# Udacity Project – Document Assistant

This repository is based on the starter template provided by Udacity for the LangChain/LangGraph course project.

## What I implemented
My original contributions are primarily the parts marked with `#TODO` in the codebase (as required by the project instructions). I also applied small integration fixes (imports, typing, and minor wiring changes) to ensure the project runs end-to-end as expected.

## Tooling
I used VS Code GitHub Copilot (non-agent mode) for code completion and refactoring suggestions. Copilot did not generate the full solution; I reviewed, edited, and implemented all final changes myself in accordance with Udacity’s Honor Code.

## Source
Udacity course project: [LangChain Agentic AI Fundamentals](https://www.udacity.com/enrollment/cd14639)

# Example Workflow Transcript

## Example 1: Q&A Intent
**User Input:**
```
"What is the total amount in invoice INV-002?"
```

**Agent Flow:**
1. `classify_intent` → classifies as "qa" (confidence: 0.85)
2. `qa_agent` →
   - Calls `document_search(query="INV-002", search_type="keyword")`
   - Calls `document_reader(doc_id="INV-002")`
   - Extracts: Total Due: $69,300
3. `update_memory` → Stores summary and active document ID
4. **Response:** "The total amount in invoice INV-002 is $69,300. (sources: INV-002)"

---

## Example 2: Summarization Intent
**User Input:**
```
"Summarize all contracts in the system"
```

**Agent Flow:**
1. `classify_intent` → classifies as "summarization" (confidence: 0.88)
2. `summarization_agent` →
   - Calls `document_search(search_type="type", doc_type="contract")`
   - Returns: CON-001
   - Calls `document_reader(doc_id="CON-001")`
   - Generates summary with key points (parties, duration, value)
3. `update_memory` → Stores summary and active document ID
4. **Response:** "Contract CON-001 is a 12-month service agreement between DocDacity Solutions Inc. and Healthcare Partners LLC with a total value of $180,000. Key services include platform access, 24/7 support, and monthly analytics. Either party can terminate with 60 days notice. (sources: CON-001)"

---

## Example 3: Calculation Intent
**User Input:**
```
"What is the total invoice amount for all invoices over $50,000?"
```

**Agent Flow:**
1. `classify_intent` → classifies as "calculation" (confidence: 0.92)
2. `calculation_agent` →
   - Calls `document_search(search_type="amount", comparison="over", amount=50000)`
   - Returns: INV-002 ($69,300), INV-003 ($214,500)
   - Constructs expression: "69300 + 214500"
   - Calls `calculator(expression="69300 + 214500")`
   - Gets result: "283800"
3. `update_memory` → Stores summary and active document IDs
4. **Response:** "The total of all invoices over $50,000 is $283,800. (sources: INV-002, INV-003)"

---

# State and Memory Management

## Checkpointer + Thread ID
- **InMemorySaver** checkpoint is created in [`create_workflow()`](src/agent.py) and persists workflow state
- Each user session has a unique `thread_id` set to `session_id` in the config dictionary passed to `workflow.invoke()`
- When the same session resumes, `workflow.get_state(config)` retrieves the prior state from the checkpointer

## Conversation Summary and Active Documents
- **`conversation_summary`** (string): Updated by [`update_memory`](src/agent.py) after each turn; stores high-level recap of conversation
- **`active_documents`** (list of doc IDs): Updated by [`update_memory`](src/agent.py); tracks which documents are relevant to the current conversation
- Both fields persist in the [`SessionState`](src/schemas.py) object and are saved to disk via [`_save_session()`](src/assistant.py)
- On next interaction, they are loaded via [`_load_session()`](src/assistant.py) and passed to the workflow in `initial_state`

## Session Persistence
- Session files stored in `./sessions/<session_id>.json` on disk
- In-memory checkpointer (InMemorySaver) stores LangGraph state during workflow execution
- For multi-turn conversations, the checkpointer allows `workflow.get_state()` to retrieve prior messages and state without re-initialization

## Structured Outputs  
- [`UserIntent`](src/schemas.py), [`AnswerResponse`](src/schemas.py), [`SummarizationResponse`](src/schemas.py), and [`CalculationResponse`](src/schemas.py) are Pydantic models
- Each agent node calls `llm.with_structured_output(ResponseSchema)` to enforce schema validation
- If LLM response doesn't match schema, LangChain automatically retries with schema hints
- Structured outputs flow through state as typed objects, preventing lossy string parsing
