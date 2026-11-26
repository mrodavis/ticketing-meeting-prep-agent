#!/usr/bin/env node
import { App } from "aws-cdk-lib";
import { MeetingPrepStack } from "../lib/meeting-prep-stack";

const app = new App();
const ctx = (key: string): string | undefined => app.node.tryGetContext(key) || undefined;

new MeetingPrepStack(app, ctx("stackName") ?? "MeetingPrep", {
  env: { account: process.env.CDK_DEFAULT_ACCOUNT, region: process.env.CDK_DEFAULT_REGION ?? "us-east-1" },
  modelId: ctx("modelId") ?? "us.anthropic.claude-sonnet-5",
  foundationModelId: ctx("foundationModelId") ?? "anthropic.claude-sonnet-5",
  publicLoadBalancer: ctx("publicLoadBalancer") === "true",
  allowedCidr: ctx("allowedCidr"),
  certificateArn: ctx("certificateArn"),
  alarmEmail: ctx("alarmEmail"),
  desiredCount: ctx("desiredCount") ? Number(ctx("desiredCount")) : undefined,
});
