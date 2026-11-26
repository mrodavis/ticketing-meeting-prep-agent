import { App } from "aws-cdk-lib";
import { Match, Template } from "aws-cdk-lib/assertions";
import { MeetingPrepProps, MeetingPrepStack } from "../lib/meeting-prep-stack";

const BASE: MeetingPrepProps = {
  env: { account: "111122223333", region: "us-east-1" },
  modelId: "us.anthropic.claude-sonnet-5",
  foundationModelId: "anthropic.claude-sonnet-5",
};

function synth(overrides: Partial<MeetingPrepProps> = {}): Template {
  return Template.fromStack(new MeetingPrepStack(new App(), "Test", { ...BASE, ...overrides }));
}

const template = synth();

function container(t: Template) {
  const defs = t.findResources("AWS::ECS::TaskDefinition");
  return Object.values(defs)[0].Properties.ContainerDefinitions[0];
}

describe("Bedrock access", () => {
  test("the task role can invoke one inference profile, and the model only through it", () => {
    template.hasResourceProperties("AWS::IAM::Policy", {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Sid: "InvokeThroughInferenceProfile",
            Action: "bedrock:InvokeModel",
            Resource: "arn:aws:bedrock:us-east-1:111122223333:inference-profile/us.anthropic.claude-sonnet-5",
          }),
          Match.objectLike({
            Sid: "FoundationModelOnlyViaThatProfile",
            Resource: "arn:aws:bedrock:*::foundation-model/anthropic.claude-sonnet-5",
            Condition: {
              StringLike: { "bedrock:InferenceProfileArn": "arn:aws:bedrock:us-east-1:111122223333:inference-profile/us.anthropic.claude-sonnet-5" },
            },
          }),
          Match.objectLike({ Sid: "ApplyOurGuardrail", Action: "bedrock:ApplyGuardrail" }),
        ]),
      },
    });
  });

  test("no policy grants bedrock actions with a wildcard action or on every resource", () => {
    for (const policy of Object.values(template.findResources("AWS::IAM::Policy"))) {
      for (const s of policy.Properties.PolicyDocument.Statement) {
        const acts = ([] as string[]).concat(s.Action);
        if (acts.some((a) => a.startsWith("bedrock:"))) {
          expect(acts).not.toContain("bedrock:*");
          expect(s.Resource).not.toEqual("*");
        }
      }
    }
  });

  test("a guardrail with a prompt-attack filter is created and passed to the app", () => {
    template.hasResourceProperties("AWS::Bedrock::Guardrail", {
      ContentPolicyConfig: { FiltersConfig: Match.arrayWith([{ Type: "PROMPT_ATTACK", InputStrength: "HIGH", OutputStrength: "NONE" }]) },
      TopicPolicyConfig: { TopicsConfig: [Match.objectLike({ Name: "ContractCommitments", Type: "DENY" })] },
    });
    const env = container(template).Environment;
    expect(env).toContainEqual({ Name: "BEDROCK_GUARDRAIL_ID", Value: { "Fn::GetAtt": ["Guardrail", "GuardrailId"] } });
    expect(env).toContainEqual({ Name: "BEDROCK_GUARDRAIL_VERSION", Value: { "Fn::GetAtt": ["GuardrailVersion", "Version"] } });
  });
});

describe("secrets and database", () => {
  test("credentials arrive as ECS secrets, never as plain environment values", () => {
    const c = container(template);
    expect(c.Secrets.map((s: { Name: string }) => s.Name).sort()).toEqual(["PGDATABASE", "PGHOST", "PGPASSWORD", "PGPORT", "PGUSER", "STAFF_TOKEN"]);
    const envNames = c.Environment.map((e: { Name: string }) => e.Name);
    for (const sensitive of ["PGPASSWORD", "STAFF_TOKEN", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]) {
      expect(envNames).not.toContain(sensitive);
    }
    expect(c.Environment).toContainEqual({ Name: "CHECKPOINT_BACKEND", Value: "postgres" });
    expect(c.Environment).toContainEqual({ Name: "PGSSLMODE", Value: "require" });
  });

  test("the database is encrypted, private, backed up and snapshotted on delete", () => {
    template.hasResource("AWS::RDS::DBInstance", {
      Properties: Match.objectLike({ StorageEncrypted: true, PubliclyAccessible: false, BackupRetentionPeriod: 7 }),
      DeletionPolicy: "Snapshot",
    });
    template.hasResourceProperties("AWS::RDS::DBParameterGroup", { Parameters: { "rds.force_ssl": "1" } });
  });

  test("only the service's tasks can reach the database port", () => {
    const ingress = Object.values(template.findResources("AWS::EC2::SecurityGroupIngress")).filter(
      (r) => JSON.stringify(r.Properties.GroupId).includes("DbSg"),
    );
    expect(ingress).toHaveLength(1);
    expect(JSON.stringify(ingress[0].Properties.SourceSecurityGroupId)).toContain("ServiceSecurityGroup");
    expect(ingress[0].Properties.CidrIp).toBeUndefined();
  });
});

describe("service and network", () => {
  test("internal load balancer by default, reachable only from inside the VPC", () => {
    template.hasResourceProperties("AWS::ElasticLoadBalancingV2::LoadBalancer", { Scheme: "internal" });
    template.hasResourceProperties("AWS::EC2::SecurityGroup", {
      SecurityGroupIngress: [Match.objectLike({ CidrIp: { "Fn::GetAtt": [Match.stringLikeRegexp("Vpc"), "CidrBlock"] }, FromPort: 80 })],
    });
  });

  test("public mode needs an allowed range, and HTTPS redirects HTTP when a certificate is given", () => {
    expect(() => synth({ publicLoadBalancer: true })).toThrow(/allowedCidr/);
    const pub = synth({
      publicLoadBalancer: true,
      allowedCidr: "203.0.113.10/32",
      certificateArn: "arn:aws:acm:us-east-1:111122223333:certificate/abc",
    });
    pub.hasResourceProperties("AWS::ElasticLoadBalancingV2::LoadBalancer", { Scheme: "internet-facing" });
    pub.hasResourceProperties("AWS::ElasticLoadBalancingV2::Listener", { Protocol: "HTTPS", Port: 443 });
    pub.hasResourceProperties("AWS::ElasticLoadBalancingV2::Listener", {
      Protocol: "HTTP",
      DefaultActions: [Match.objectLike({ Type: "redirect" })],
    });
    const lbSg = Object.values(pub.findResources("AWS::EC2::SecurityGroup")).find((r) => r.Properties.GroupDescription?.includes("ServiceLB"));
    const cidrs = (lbSg!.Properties.SecurityGroupIngress as Array<{ CidrIp: string }>).map((i) => i.CidrIp);
    expect(new Set(cidrs)).toEqual(new Set(["203.0.113.10/32"]));
  });

  test("tasks run in private subnets with health checks and deployment rollback", () => {
    template.hasResourceProperties("AWS::ECS::Service", {
      DesiredCount: 2,
      DeploymentConfiguration: Match.objectLike({ DeploymentCircuitBreaker: { Enable: true, Rollback: true } }),
      NetworkConfiguration: { AwsvpcConfiguration: Match.objectLike({ AssignPublicIp: "DISABLED" }) },
    });
    template.hasResourceProperties("AWS::ElasticLoadBalancingV2::TargetGroup", { HealthCheckPath: "/healthz" });
  });
});

describe("observability", () => {
  test("alarms cover errors, latency, health, fallbacks, model failures and the database, all notifying SNS", () => {
    const alarms = Object.values(template.findResources("AWS::CloudWatch::Alarm"));
    expect(alarms).toHaveLength(8);
    for (const a of alarms) expect(a.Properties.AlarmActions).toHaveLength(1);
    template.hasResourceProperties("AWS::CloudWatch::Alarm", { Namespace: "MeetingPrep", MetricName: "Fallback" });
    template.hasResourceProperties("AWS::CloudWatch::Alarm", { Namespace: "MeetingPrep", MetricName: "LlmCallFailures" });
  });

  test("the app emits EMF metrics and a dashboard reads them", () => {
    expect(container(template).Environment).toContainEqual({ Name: "EMF_ENABLED", Value: "1" });
    const body = JSON.stringify(template.findResources("AWS::CloudWatch::Dashboard"));
    for (const metric of ["BriefRuns", "Fallback", "InputTokens", "DecisionApprove"]) expect(body).toContain(metric);
  });
});
