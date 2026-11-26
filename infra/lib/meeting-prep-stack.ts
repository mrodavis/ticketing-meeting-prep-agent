import * as path from "path";
import { CfnOutput, Duration, RemovalPolicy, Stack, StackProps } from "aws-cdk-lib";
import * as bedrock from "aws-cdk-lib/aws-bedrock";
import * as acm from "aws-cdk-lib/aws-certificatemanager";
import * as cloudwatch from "aws-cdk-lib/aws-cloudwatch";
import * as actions from "aws-cdk-lib/aws-cloudwatch-actions";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as ecs from "aws-cdk-lib/aws-ecs";
import * as patterns from "aws-cdk-lib/aws-ecs-patterns";
import * as elbv2 from "aws-cdk-lib/aws-elasticloadbalancingv2";
import * as iam from "aws-cdk-lib/aws-iam";
import * as logs from "aws-cdk-lib/aws-logs";
import * as rds from "aws-cdk-lib/aws-rds";
import * as secretsmanager from "aws-cdk-lib/aws-secretsmanager";
import * as sns from "aws-cdk-lib/aws-sns";
import * as subs from "aws-cdk-lib/aws-sns-subscriptions";
import { Construct } from "constructs";

export interface MeetingPrepProps extends StackProps {
  /** Bedrock inference profile the app calls, e.g. us.anthropic.claude-sonnet-5 */
  readonly modelId: string;
  /** Foundation model behind that profile, e.g. anthropic.claude-sonnet-5 */
  readonly foundationModelId: string;
  /** Internal by default: a staff tool reached over VPN or Direct Connect. */
  readonly publicLoadBalancer?: boolean;
  /** Who may reach the API. Required when the load balancer is public; defaults to the VPC's own range. */
  readonly allowedCidr?: string;
  /** Adds an HTTPS listener and redirects HTTP to it. */
  readonly certificateArn?: string;
  readonly alarmEmail?: string;
  readonly desiredCount?: number;
}

export const NAMESPACE = "MeetingPrep";
const SERVICE_DIM = { Service: "meeting-prep" };

export class MeetingPrepStack extends Stack {
  constructor(scope: Construct, id: string, props: MeetingPrepProps) {
    super(scope, id, props);
    if (props.publicLoadBalancer && !props.allowedCidr) {
      throw new Error("A public load balancer needs allowedCidr (for example -c allowedCidr=203.0.113.10/32)");
    }

    // ---- Network: public subnets for the ALB, private for tasks, isolated for the database.
    const vpc = new ec2.Vpc(this, "Vpc", {
      maxAzs: 2,
      natGateways: 1, // one NAT keeps cost down; use one per AZ for production
      subnetConfiguration: [
        { name: "public", subnetType: ec2.SubnetType.PUBLIC, cidrMask: 24 },
        { name: "app", subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS, cidrMask: 24 },
        { name: "db", subnetType: ec2.SubnetType.PRIVATE_ISOLATED, cidrMask: 24 },
      ],
    });
    // Bedrock and Secrets Manager traffic stays on the AWS network instead of going through the NAT.
    vpc.addInterfaceEndpoint("BedrockRuntime", { service: ec2.InterfaceVpcEndpointAwsService.BEDROCK_RUNTIME });
    vpc.addInterfaceEndpoint("SecretsManager", { service: ec2.InterfaceVpcEndpointAwsService.SECRETS_MANAGER });

    // ---- Postgres for LangGraph checkpoints, so any task can resume a paused run.
    const dbSg = new ec2.SecurityGroup(this, "DbSg", { vpc, description: "Checkpoint database", allowAllOutbound: false });
    const db = new rds.DatabaseInstance(this, "CheckpointDb", {
      engine: rds.DatabaseInstanceEngine.postgres({ version: rds.PostgresEngineVersion.VER_16 }),
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.T4G, ec2.InstanceSize.MICRO),
      vpc,
      vpcSubnets: { subnetType: ec2.SubnetType.PRIVATE_ISOLATED },
      securityGroups: [dbSg],
      credentials: rds.Credentials.fromGeneratedSecret("meeting_prep"),
      databaseName: "meeting_prep",
      allocatedStorage: 20,
      maxAllocatedStorage: 50,
      storageEncrypted: true,
      publiclyAccessible: false,
      backupRetention: Duration.days(7),
      deletionProtection: false, // demo stack; turn on for real use
      removalPolicy: RemovalPolicy.SNAPSHOT,
      cloudwatchLogsExports: ["postgresql"],
      parameters: { "rds.force_ssl": "1" },
    });
    const dbSecret = db.secret!;

    const staffToken = new secretsmanager.Secret(this, "StaffToken", {
      description: "Bearer token for the meeting prep API",
      generateSecretString: { passwordLength: 48, excludePunctuation: true },
    });

    // ---- Bedrock Guardrail on every model call.
    const guardrail = new bedrock.CfnGuardrail(this, "Guardrail", {
      name: `${this.stackName}-guardrail`,
      description: "Blocks prompt attacks and commitments on contract terms; masks card and Social Security numbers.",
      blockedInputMessaging: "This request was blocked by the meeting prep guardrail.",
      blockedOutputsMessaging: "The draft was blocked by the meeting prep guardrail.",
      contentPolicyConfig: {
        filtersConfig: [
          { type: "PROMPT_ATTACK", inputStrength: "HIGH", outputStrength: "NONE" },
          { type: "MISCONDUCT", inputStrength: "MEDIUM", outputStrength: "MEDIUM" },
          { type: "INSULTS", inputStrength: "MEDIUM", outputStrength: "MEDIUM" },
          { type: "HATE", inputStrength: "MEDIUM", outputStrength: "MEDIUM" },
        ],
      },
      topicPolicyConfig: {
        topicsConfig: [
          {
            name: "ContractCommitments",
            type: "DENY",
            definition: "Promising discounts, credits, refunds or contract terms on behalf of the venue.",
            examples: ["We will give you 40% off next season.", "I guarantee your renewal price will not go up."],
          },
        ],
      },
      sensitiveInformationPolicyConfig: {
        piiEntitiesConfig: [
          { type: "CREDIT_DEBIT_CARD_NUMBER", action: "ANONYMIZE" },
          { type: "US_SOCIAL_SECURITY_NUMBER", action: "ANONYMIZE" },
        ],
      },
    });
    const guardrailVersion = new bedrock.CfnGuardrailVersion(this, "GuardrailVersion", {
      guardrailIdentifier: guardrail.attrGuardrailId,
      description: "Deployed with the stack",
    });

    // ---- ECS on Fargate behind an ALB.
    const cluster = new ecs.Cluster(this, "Cluster", { vpc, containerInsightsV2: ecs.ContainerInsights.ENHANCED });
    const logGroup = new logs.LogGroup(this, "AppLogs", { retention: logs.RetentionDays.ONE_MONTH, removalPolicy: RemovalPolicy.DESTROY });

    const app = new patterns.ApplicationLoadBalancedFargateService(this, "Service", {
      cluster,
      cpu: 512,
      memoryLimitMiB: 1024,
      desiredCount: props.desiredCount ?? 2,
      publicLoadBalancer: props.publicLoadBalancer ?? false,
      openListener: false,
      taskSubnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
      circuitBreaker: { enable: true, rollback: true },
      minHealthyPercent: 100,
      maxHealthyPercent: 200,
      runtimePlatform: { cpuArchitecture: ecs.CpuArchitecture.X86_64, operatingSystemFamily: ecs.OperatingSystemFamily.LINUX },
      ...(props.certificateArn
        ? {
            certificate: acm.Certificate.fromCertificateArn(this, "Cert", props.certificateArn),
            protocol: elbv2.ApplicationProtocol.HTTPS,
            redirectHTTP: true,
            sslPolicy: elbv2.SslPolicy.RECOMMENDED_TLS,
          }
        : {}),
      taskImageOptions: {
        image: ecs.ContainerImage.fromAsset(path.join(__dirname, "..", "..")),
        containerPort: 8000,
        logDriver: ecs.LogDrivers.awsLogs({ streamPrefix: "app", logGroup }),
        environment: {
          LLM_MODE: "bedrock",
          AWS_REGION: this.region,
          BEDROCK_MODEL_ID: props.modelId,
          BEDROCK_GUARDRAIL_ID: guardrail.attrGuardrailId,
          BEDROCK_GUARDRAIL_VERSION: guardrailVersion.attrVersion,
          CHECKPOINT_BACKEND: "postgres",
          PGSSLMODE: "require",
          EMF_ENABLED: "1",
        },
        secrets: {
          STAFF_TOKEN: ecs.Secret.fromSecretsManager(staffToken),
          PGHOST: ecs.Secret.fromSecretsManager(dbSecret, "host"),
          PGPORT: ecs.Secret.fromSecretsManager(dbSecret, "port"),
          PGUSER: ecs.Secret.fromSecretsManager(dbSecret, "username"),
          PGPASSWORD: ecs.Secret.fromSecretsManager(dbSecret, "password"),
          PGDATABASE: ecs.Secret.fromSecretsManager(dbSecret, "dbname"),
        },
      },
    });
    app.targetGroup.configureHealthCheck({ path: "/healthz", healthyHttpCodes: "200", interval: Duration.seconds(15) });
    app.targetGroup.setAttribute("deregistration_delay.timeout_seconds", "30");
    // Brief generation calls a model and can take a while.
    app.loadBalancer.setAttribute("idle_timeout.timeout_seconds", "120");

    const lbSg = app.loadBalancer.connections;
    // Internal by default: only addresses inside the VPC (or allowedCidr, such as a VPN range) reach the API.
    const allowed = ec2.Peer.ipv4(props.allowedCidr ?? vpc.vpcCidrBlock);
    lbSg.allowFrom(allowed, ec2.Port.tcp(props.certificateArn ? 443 : 80), "Who may reach the API");
    if (props.certificateArn) lbSg.allowFrom(allowed, ec2.Port.tcp(80), "HTTP redirect to HTTPS");

    db.connections.allowDefaultPortFrom(app.service, "Tasks reach the checkpoint database");

    const scaling = app.service.autoScaleTaskCount({ minCapacity: props.desiredCount ?? 2, maxCapacity: 4 });
    scaling.scaleOnCpuUtilization("Cpu", { targetUtilizationPercent: 60 });

    // ---- Least-privilege Bedrock access for the task role.
    const profileArn = `arn:aws:bedrock:${this.region}:${this.account}:inference-profile/${props.modelId}`;
    const role = app.taskDefinition.taskRole;
    role.addToPrincipalPolicy(
      new iam.PolicyStatement({ sid: "InvokeThroughInferenceProfile", actions: ["bedrock:InvokeModel"], resources: [profileArn] }),
    );
    role.addToPrincipalPolicy(
      new iam.PolicyStatement({
        sid: "FoundationModelOnlyViaThatProfile",
        actions: ["bedrock:InvokeModel"],
        resources: [`arn:aws:bedrock:*::foundation-model/${props.foundationModelId}`],
        conditions: { StringLike: { "bedrock:InferenceProfileArn": profileArn } },
      }),
    );
    role.addToPrincipalPolicy(
      new iam.PolicyStatement({ sid: "ApplyOurGuardrail", actions: ["bedrock:ApplyGuardrail"], resources: [guardrail.attrGuardrailArn] }),
    );

    // ---- Alarms.
    const topic = new sns.Topic(this, "Alarms", { displayName: "Meeting prep alarms" });
    if (props.alarmEmail) topic.addSubscription(new subs.EmailSubscription(props.alarmEmail));
    const notify = new actions.SnsAction(topic);
    const appMetric = (name: string, statistic = "Sum", period = Duration.minutes(5)) =>
      new cloudwatch.Metric({ namespace: NAMESPACE, metricName: name, dimensionsMap: SERVICE_DIM, statistic, period });

    const lb = app.loadBalancer.metrics;
    const tg = app.targetGroup.metrics;
    const alarms: Array<[string, cloudwatch.IMetric, number, string]> = [
      ["Api5xx", tg.httpCodeTarget(elbv2.HttpCodeTarget.TARGET_5XX_COUNT, { period: Duration.minutes(5) }), 5, "API returned 5xx responses"],
      ["Lb5xx", lb.httpCodeElb(elbv2.HttpCodeElb.ELB_5XX_COUNT, { period: Duration.minutes(5) }), 5, "Load balancer returned 5xx (no healthy target, timeouts)"],
      ["SlowBriefs", tg.targetResponseTime({ statistic: "p95", period: Duration.minutes(5) }), 30, "p95 response time over 30 seconds"],
      ["UnhealthyTasks", tg.unhealthyHostCount({ period: Duration.minutes(1) }), 1, "A task is failing its health check"],
      ["Fallbacks", appMetric("Fallback", "Sum", Duration.minutes(15)), 3, "Briefs falling back to facts-only (model or checks failing)"],
      ["ModelCallFailures", appMetric("LlmCallFailures", "Sum", Duration.minutes(15)), 5, "Bedrock calls failing (throttling, access, guardrail)"],
      ["DbCpu", db.metricCPUUtilization({ period: Duration.minutes(5) }), 80, "Checkpoint database CPU over 80%"],
    ];
    for (const [id, metric, threshold, description] of alarms) {
      new cloudwatch.Alarm(this, id, {
        metric,
        threshold,
        evaluationPeriods: id === "UnhealthyTasks" ? 3 : 1,
        comparisonOperator: cloudwatch.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
        treatMissingData: cloudwatch.TreatMissingData.NOT_BREACHING,
        alarmDescription: description,
      }).addAlarmAction(notify);
    }
    new cloudwatch.Alarm(this, "DbStorageLow", {
      metric: db.metricFreeStorageSpace({ period: Duration.minutes(5) }),
      threshold: 2 * 1024 ** 3,
      evaluationPeriods: 1,
      comparisonOperator: cloudwatch.ComparisonOperator.LESS_THAN_THRESHOLD,
      alarmDescription: "Checkpoint database under 2 GiB free",
    }).addAlarmAction(notify);

    // ---- Dashboard.
    const dashboard = new cloudwatch.Dashboard(this, "Dashboard", { dashboardName: `meeting-prep-${this.stackName}` });
    dashboard.addWidgets(
      new cloudwatch.GraphWidget({ title: "Requests and 5xx", left: [lb.requestCount()], right: [tg.httpCodeTarget(elbv2.HttpCodeTarget.TARGET_5XX_COUNT)], width: 12 }),
      new cloudwatch.GraphWidget({
        title: "Response time",
        left: [tg.targetResponseTime({ statistic: "p50" }), tg.targetResponseTime({ statistic: "p95" })],
        width: 12,
      }),
      new cloudwatch.GraphWidget({
        title: "Brief runs",
        left: [appMetric("BriefRuns"), appMetric("FirstDraftPassed"), appMetric("Fallback")],
        width: 12,
      }),
      new cloudwatch.GraphWidget({ title: "Reliability", left: [appMetric("ToolRetries"), appMetric("LlmCallFailures"), appMetric("DataGaps")], width: 12 }),
      new cloudwatch.GraphWidget({ title: "Tokens", left: [appMetric("InputTokens"), appMetric("OutputTokens")], width: 12 }),
      new cloudwatch.GraphWidget({
        title: "Rep decisions",
        left: [appMetric("DecisionApprove"), appMetric("DecisionEdit"), appMetric("DecisionReject")],
        width: 12,
      }),
      new cloudwatch.GraphWidget({ title: "Tasks", left: [app.service.metricCpuUtilization(), app.service.metricMemoryUtilization()], width: 12 }),
      new cloudwatch.GraphWidget({ title: "Database", left: [db.metricCPUUtilization()], right: [db.metricDatabaseConnections()], width: 12 }),
    );

    new CfnOutput(this, "ApiUrl", { value: `${props.certificateArn ? "https" : "http"}://${app.loadBalancer.loadBalancerDnsName}` });
    new CfnOutput(this, "StaffTokenSecretArn", { value: staffToken.secretArn });
    new CfnOutput(this, "DashboardName", { value: dashboard.dashboardName });
  }
}
