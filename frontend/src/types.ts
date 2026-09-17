// frontend/src/types.ts
//
// TypeScript mirrors of the API's response shapes.
//
// These are written by hand against the FastAPI schemas rather than generated, and the
// tradeoff is deliberate: the set is small, and hand-written types can carry the unions the
// backend actually guarantees (`StageName`, `EventStatus`) which a generator would flatten to
// `string`. The rule that keeps them honest is that every type here corresponds to exactly one
// `as_dict()` in `multi_agent_generator/core/models.py` or one Pydantic model in
// `backend/schemas.py` - if a field is optional here it is because it is optional there.

/** Pipeline stages. These strings are part of the API contract (`core.models.Stage`). */
export type StageName =
  | "analysis"
  | "framework_selection"
  | "planning"
  | "generation"
  | "review"
  | "improvement"
  | "repair"
  | "test"
  | "execution"
  | "validation";

export type EventStatus = "started" | "succeeded" | "failed" | "info" | "warning";

export type ProjectStatus =
  | "draft"
  | "analyzing"
  | "generating"
  | "reviewing"
  | "improving"
  | "testing"
  | "ready"
  | "failed";

/**
 * The single error shape every endpoint returns.
 *
 * `action` is the field the UI should show most prominently after `message` - it is the
 * sentence the user can act on ("Set HF_TOKEN in your .env file"), which is the whole reason
 * the backend carries it separately instead of concatenating it into the message.
 */
export interface ApiError {
  code: string;
  message: string;
  action?: string | null;
  detail?: string | null;
  context?: Record<string, unknown>;
}

export interface PipelineEvent {
  stage: StageName;
  status: EventStatus;
  level: string;
  message: string;
  timestamp: string;
  duration_s: number | null;
  project_id: string | null;
  run_id: string | null;
  iteration: number | null;
  error: ApiError | null;
  data: Record<string, unknown>;
}

export interface GeneratedFile {
  path: string;
  content: string;
  language: string;
  lines: number;
  is_entrypoint: boolean;
  description: string;
}

export interface GeneratedProject {
  framework: string;
  provider: string;
  model: string;
  entrypoint: string | null;
  dependencies: string[];
  run_instructions: string;
  notes: string[];
  config: Record<string, unknown>;
  file_count: number;
  total_lines: number;
  files: GeneratedFile[];
}

export interface RequirementAnalysis {
  requirement: string;
  summary: string;
  goals: string[];
  suggested_roles: string[];
  suggested_tools: string[];
  needs_sequential_steps: boolean;
  needs_branching: boolean;
  needs_delegation: boolean;
  needs_state: boolean;
  needs_human_input: boolean;
  complexity: string;
  risks: string[];
  /** False when the offline heuristic produced this rather than a model. */
  from_model: boolean;
}

export interface FrameworkChoice {
  framework: string;
  reason: string;
  confidence: number;
  user_specified: boolean;
  alternatives: { framework?: string; reason?: string }[];
}

export interface ArchitectureSummary {
  project_name: string;
  tier: string;
  framework: string;
  provider: string;
  model: string;
  package: string;
  agents: number;
  tools: number;
  workflows: number;
  workflow_steps: number;
  services: number;
  tests: number;
  files: number;
  python_files: number;
  entrypoint: string | null;
  runtime: string;
  dependencies: number;
  env_vars: string[];
}

export interface ArchitectureAgent {
  name: string;
  label: string;
  role: string;
  goal: string;
  backstory: string;
  responsibilities: string[];
  tools: string[];
  model: string;
  allow_delegation: boolean;
  inputs: string;
  outputs: string;
  module: string;
  factory: string;
}

export interface ArchitectureTool {
  name: string;
  label: string;
  purpose: string;
  class_name: string;
  parameters: Record<string, string>;
  requires_env: string[];
  dependencies: string[];
  module: string;
  implemented: boolean;
}

export interface ArchitectureWorkflowNode {
  id: string;
  kind: string;
  label: string;
  description: string;
  agent: string | null;
  tool: string | null;
  inputs: string;
  outputs: string;
}

export interface ArchitectureWorkflowEdge {
  source: string;
  target: string;
  condition: string | null;
  label: string;
  conditional: boolean;
}

export interface ArchitectureWorkflow {
  name: string;
  kind: string;
  description: string;
  nodes: ArchitectureWorkflowNode[];
  edges: ArchitectureWorkflowEdge[];
  entry: string;
  exits: string[];
  state_fields: string[];
  module: string;
}

export interface ArchitectureFile {
  path: string;
  kind: string;
  purpose: string;
  provides: string[];
  depends_on: string[];
  owner: string | null;
  module: string | null;
  is_entrypoint: boolean;
}

export interface ArchitecturePlan {
  project_name: string;
  framework: string;
  provider: string;
  model: string;
  requirement: string;
  description: string;
  tier: string;
  package: string;
  agents: ArchitectureAgent[];
  tools: ArchitectureTool[];
  workflows: ArchitectureWorkflow[];
  files: ArchitectureFile[];
  dependencies: string[];
  env_vars: { name: string; required: boolean; purpose: string; example: string }[];
  entrypoint: string | null;
  notes: string[];
  capabilities: string[];
  summary: ArchitectureSummary;
}

export type Severity = "blocker" | "major" | "minor" | "info";

export interface ReviewIssue {
  category: string;
  severity: Severity;
  message: string;
  file: string | null;
  line: number | null;
  suggestion: string;
}

export interface ReviewResult {
  score: number;
  passed: boolean;
  summary: string;
  issues: ReviewIssue[];
  required_changes: string[];
  dimension_scores: Record<string, number>;
  blocker_count: number;
  from_model: boolean;
}

export interface TestReport {
  status: "passed" | "failed" | "error" | "skipped" | "timeout";
  /** True only when the suite ran and every test passed. Never derive this locally. */
  ok: boolean;
  exit_code: number | null;
  timed_out: boolean;
  duration_s: number;
  /** False when the suite never executed. A skipped suite is not a green suite. */
  ran: boolean;
  counts: { passed: number; failed: number; skipped: number; errors: number };
  failed_tests: string[];
  stdout: string;
  stderr: string;
  unavailable_reason: string | null;
  /**
   * Context that is not a verdict - most usefully, requirement lines the sandbox refused to
   * install. Shown alongside a failure rather than instead of it: a refused
   * `git+https://...` line is the reason an import test failed, and without it the user sees
   * only an unexplained ModuleNotFoundError.
   */
  notes: string[];
}

export interface IterationRecord {
  index: number;
  review: ReviewResult | null;
  tests: TestReport | null;
  changes_applied: string[];
  duration_s: number;
}

export interface PipelineResult {
  requirement: string;
  status: ProjectStatus;
  /** True only when review passed AND the generated tests ran AND passed. */
  succeeded: boolean;
  /** Why `succeeded` is false. Empty when it is true. */
  unmet_criteria: string[];
  analysis: RequirementAnalysis | null;
  framework_choice: FrameworkChoice | null;
  manifest: ArchitecturePlan | null;
  architecture: ArchitectureSummary | null;
  project: GeneratedProject | null;
  iterations: IterationRecord[];
  review: ReviewResult | null;
  tests: TestReport | null;
  error: ApiError | null;
  started_at: string;
  completed_at: string | null;
  duration_s: number;
  events: PipelineEvent[];
}

export interface RunSnapshot {
  run_id: string;
  project_id: string | null;
  requirement: string;
  status: "queued" | "running" | "done" | "error";
  finished: boolean;
  started_at: number;
  finished_at: number | null;
  duration_s: number;
  event_count: number;
  events: PipelineEvent[];
  result: PipelineResult | null;
  error: ApiError | null;
}

export interface CreateProjectResponse {
  project_id: string | null;
  run_id: string;
  status: string;
  /** False when the server booted without a database; the run still executes. */
  persisted: boolean;
  events_url: string;
  run_url: string;
}

export interface ProjectSummary {
  id: string;
  name: string;
  requirement: string;
  framework: string | null;
  provider: string | null;
  model: string | null;
  status: ProjectStatus;
  created_at: string;
  updated_at: string;
}

export interface ProjectRecord extends ProjectSummary {
  result: PipelineResult | null;
}

export interface FrameworkInfo {
  name: string;
  label: string;
  description: string;
  strengths: string[];
  best_for: string[];
  installed: boolean;
  install_command: string | null;
}

export interface ProviderInfo {
  name: string;
  label: string;
  description: string;
  default_model: string | null;
  runtime_model: string | null;
  credential_env: string[];
  /** Whether a credential for it is currently resolvable. */
  configured: boolean;
  /** Whether its Python packages are importable. */
  installed: boolean;
  install_command: string | null;
  local: boolean;
  notes: string | null;
}

/** Note the absence of an `api_key` field. The read model structurally cannot return one. */
export interface LLMSettings {
  provider: string;
  model: string | null;
  base_url: string | null;
  temperature: number | null;
  max_tokens: number | null;
  timeout: number | null;
  top_p: number | null;
  extra: Record<string, unknown>;
  credential_configured: boolean;
  /** Last four characters, so a user can tell which key is saved. */
  credential_hint: string | null;
  credential_from_env: boolean;
  can_store_credentials: boolean;
  source: string;
}

export interface Health {
  status: string;
  version: string;
  persistence: boolean;
  persistence_error: string | null;
  encryption: boolean;
  default_provider: string;
  credential_configured: boolean;
}

export interface ConnectionCheck {
  ok: boolean;
  provider?: string;
  model?: string;
  message: string;
  latency_s?: number | null;
  code?: string | null;
}

export interface AgentRunResult {
  status: string;
  ok: boolean;
  exit_code: number | null;
  timed_out: boolean;
  duration_s: number;
  output: string | null;
  /**
   * What the agent actually did, one entry per step, as the framework itself recorded it.
   * Empty when the project's run function returned only an answer - render "no steps
   * recorded" in that case rather than inventing a plausible-looking trace.
   */
  steps: AgentStep[];
  /** Tool invocations the run recorded. Empty means none happened, not "unknown". */
  tool_calls: AgentToolCall[];
  /** Which function in the generated project was called. Distinguishes "no run function
   * found" from "the run function failed". */
  entry: string | null;
  stdout: string;
  stderr: string;
  error: ApiError | null;
  usage: Record<string, number | null>;
  project_id?: string;
}

/**
 * One step of a run. Deliberately loose: each framework records its own shape, and the
 * runtime passes it through untouched rather than flattening it into a lowest common
 * denominator that would drop the detail worth showing.
 */
export interface AgentStep {
  agent?: string;
  action?: string;
  status?: string;
  output?: string;
  duration_s?: number;
  [key: string]: unknown;
}

export interface AgentToolCall {
  name?: string;
  args?: unknown;
  result?: unknown;
  [key: string]: unknown;
}

/** Order the progress panel renders stages in. Matches the pipeline's execution order. */
export const STAGE_ORDER: StageName[] = [
  "analysis",
  "framework_selection",
  "planning",
  "generation",
  "review",
  "improvement",
  "test",
  "execution",
  "validation",
];

export const STAGE_LABELS: Record<StageName, string> = {
  analysis: "Analyse the requirement",
  framework_selection: "Select a framework",
  planning: "Plan the architecture",
  generation: "Generate the project",
  review: "Review the code",
  improvement: "Apply improvements",
  repair: "Repair failed gates",
  test: "Run the generated tests",
  execution: "Execute the agent",
  validation: "Validate the result",
};
