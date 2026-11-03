export type TurnState = 'pending'|'running'|'completed'|'incomplete'|'failed'|'cancelled'|'interrupted'
export interface Evidence { title: string; source_url: string|null; source_locator: string; page: number|null; text: string }
export interface AgentResult { agent_type: string; intent_ids: string[]; execution_status: string; stop_reason: string; unresolved: string[]; evidence: Evidence[] }
export interface ToolActivity { name:string; state:'waiting'|'running'|'completed'|'failed'|'timed_out'|'cancelled' }
export interface Status { phase?: string; agent_type?: string; execution_status?: string; tools?:ToolActivity[]; current_step?: number; steps_remaining?: number; tool_calls_used?: number; elapsed_ms?: number; remaining_time_ms?: number; stop_reason?: string }
export interface Routing { agent_types: string[]; matched_intents: string[]; mode: string; fallback_reason: string|null }
export interface Turn { turn_id: string; message: string; state: TurnState; answer: string|null; routing: Routing|null; agent_results: AgentResult[]; status: Status[]; error_code: string|null }
export interface ConversationSummary { id: string; title: string; updated_at: string }
export interface Conversation { id: string; title: string; turns: Turn[] }
export interface Health { mode: 'demo'|'live'; prompt_version: string; dependencies: Record<string,string> }
export interface SendBody { message: string; turn_id: string }
export const terminal = (t: Turn) => !['pending','running'].includes(t.state)
