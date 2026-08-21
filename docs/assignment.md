# **Agentic Google Workspace Orchestrator \- Technical Assignment**

## **Overview**

Build an intelligent orchestrator that executes natural language queries across Gmail, Google Calendar, and Google Drive. The system must classify intent, route to appropriate services, execute operations in parallel, and synthesize coherent responses.

**Time:** 6-8 hours | **Stack:** Python, FastAPI, PostgreSQL (pgvector), Redis, OpenAI/Anthropic

---

## **Problem Statement**

Users want to interact with their Google Workspace using natural language:

"Cancel my Turkish Airlines flight"

→ Search Gmail for booking → Find calendar event → Draft cancellation email

"Prepare for tomorrow's client meeting with Acme Corp"  

→ Find calendar event → Search emails with client → Pull Drive documents

"What's on my calendar next week where john@company.com is invited?"

→ Search calendar → Filter by attendee → Return formatted list

Your job: Build the orchestration layer that makes this work.

---

## 

## **System Architecture**

User Query

    ↓

Intent Classifier (LLM)

    ↓

Query Planner (Creates execution DAG)

    ↓

Service Orchestrator (Parallel execution)

    ├→ Gmail Agent (search, read, send, draft)

    ├→ GCal Agent (search, create, update, delete)  

    └→ Drive Agent (search, read, share)

    ↓

Embedding & Search Layer (Vector DB \+ semantic search)

    ↓

Response Synthesizer (Natural language output)

---

## **Core Components**

### **1\. Intent Classifier**

Parse queries into structured intents:

{

  "services": \["gmail", "gcal"\],

  "intent": "cancel\_flight",

  "entities": {"airline": "Turkish Airlines"},

  "steps": \[

    "search\_gmail\_for\_booking",

    "find\_calendar\_event", 

    "draft\_cancellation\_email"

  \]

}

### **2\. Query Planner**

Convert intent into execution plan with dependencies:

* Parallel operations: Search Gmail \+ Calendar simultaneously  
* Sequential dependencies: Extract booking reference → Draft email  
* Fallback strategies for missing data

### **3\. Service Agents**

Each agent (Gmail/GCal/Drive) implements:

* `search()`: Semantic search using embeddings  
* `execute()`: Perform write operations (send, create, delete)  
* `get_context()`: Retrieve full content for LLM reasoning

**Gmail:** search\_emails, get\_email, send\_email, draft\_email, update\_labels  
 **GCal:** search\_events, get\_event, create\_event, update\_event, delete\_event  
 **Drive:** search\_files, get\_file, share\_file, create\_folder, move\_file

### **4\. Embedding & Search**

* Generate embeddings for emails (subject \+ body), events (title \+ description), files (name \+ content)  
* Store in pgvector with cosine similarity index  
* Hybrid search: Vector similarity \+ keyword filtering (date, sender, etc.)  
* **Target:** \<500ms query latency, Precision@5 \> 0.8

### **5\. Response Synthesizer**

Aggregate results from multiple agents and generate natural language:

"I found your Turkish Airlines booking (TK1234) in an email from Oct 15\.

✓ Calendar event "Istanbul → NYC Flight" on Nov 5 at 10:30 AM

✓ Drafted cancellation email to support@turkishairlines.com

Would you like me to send it?"

---

## **Sample Queries (Test These)**

**Single Service:**

* "What's on my calendar next week?"  
* "Find emails from sarah@company.com about the budget"  
* "Show me PDFs in Drive from last month"

**Multi-Service:**

* "Cancel my Turkish Airlines flight" (Gmail \+ GCal)  
* "Prepare for tomorrow's meeting with Acme Corp" (GCal \+ Gmail \+ Drive)  
* "Find events next week that conflict with my out-of-office doc" (GCal \+ Drive)

**Hard Cases:**

* "Move the meeting with John" (ambiguous: which John? which meeting?)  
* "That email about the proposal" (requires conversation context)  
* "Next Tuesday" (temporal reasoning \+ timezone handling)

---

## **Scaling to 1M Users**

### **Architecture**

Load Balancer

    ↓

API Servers (FastAPI) × N

    ↓

Redis (cache) \+ PostgreSQL (metadata \+ pgvector)

    ↓

Task Queue (Celery workers)

    ↓

Google APIs \+ LLM APIs

### **Key Strategies**

1. **Caching:** Redis for embeddings (1hr TTL), intent classifications, conversation context  
2. **Rate Limiting:** 100 queries/user/hour, respect Google API quotas (250 units/sec)  
3. **Async Processing:** Celery for long-running orchestrations (2-5s queries)  
4. **Pre-computation:** Background sync every 15 mins, index new emails/events/files  
5. **Sharding:** Partition by `user_id` for metadata and embeddings  
6. **Multi-region:** Deploy in US/EU/APAC, route to nearest region

### **Metrics to Monitor**

* P99 latency: \<2s  
* Cache hit rate: \>80%  
* Google API errors: \<0.1%  
* Embedding freshness: \<15min lag

---

## **Database Schema (Simplified)**

CREATE TABLE users (

    id UUID PRIMARY KEY,

    email VARCHAR(255),

    google\_access\_token TEXT,

    google\_refresh\_token TEXT

);

CREATE TABLE conversations (

    id UUID PRIMARY KEY,

    user\_id UUID REFERENCES users(id),

    query TEXT,

    intent JSONB,

    response TEXT,

    created\_at TIMESTAMP

);

CREATE TABLE gmail\_cache (

    id UUID PRIMARY KEY,

    user\_id UUID,

    email\_id VARCHAR(255),

    subject TEXT,

    body\_preview TEXT,

    embedding vector(1536),

    received\_at TIMESTAMP,

    UNIQUE(user\_id, email\_id)

);

\-- Similar tables for gcal\_cache, gdrive\_cache

CREATE INDEX ON gmail\_cache USING ivfflat (embedding vector\_cosine\_ops);

---

## **API Endpoints**

POST /api/v1/query

{

  "query": "Cancel my Turkish Airlines flight",

  "conversation\_id": "uuid"

}

→ Response: Natural language \+ actions\_taken

GET /api/v1/auth/google

→ OAuth flow

POST /api/v1/sync/trigger

→ Manually sync Gmail/GCal/Drive

GET /api/v1/sync/status

→ Last sync timestamps per service

---

## **What Makes This Hard**

1. **Intent Ambiguity:** "Move the meeting with John" → Which John? Which meeting? When?  
2. **Cross-Service Dependencies:** Failures must be handled gracefully (Gmail succeeds, Calendar fails)  
3. **Temporal Reasoning:** "Next week" depends on timezone, work week vs calendar week, context  
4. **Embedding Quality:** What to embed? How to chunk email threads? Temporal decay?  
5. **Rate Limits:** Google APIs have strict quotas—need intelligent batching and caching  
6. **Security:** Multi-tenant isolation, OAuth token refresh, audit logging

---

## **Evaluation Criteria (100 points)**

**Core Functionality (40)**

* Intent classification accuracy (10)  
* Multi-service orchestration (15)  
* Embedding & search quality: Precision@5 \> 0.8 (10)  
* Response synthesis (5)

**Architecture & Design (30)**

* System design for 1M users (10)  
* Database schema with vector indexes (5)  
* API design (5)  
* Task queue implementation (5)  
* Caching strategy (5)

**Code Quality (15)**

* Modularity (5)  
* Error handling (5)  
* Testing (5)

**Embedding Quality (10)**

* Strategy (5)  
* Performance \<500ms (3)  
* Relevance metrics (2)

**Bonus (5)**

* WebSocket real-time updates  
* Conversation context tracking  
* Conflict detection  
* Docker setup

---

## **Submission Requirements**

**Must Include:**

1. GitHub repo with README, DESIGN.md (scaling strategy), API.md  
2. Database schema with migrations \+ ER diagram  
3. 10+ sample queries with expected outputs (include edge cases)  
4. Postman collection or OpenAPI spec  
5. 5min video demo showing complex orchestration

**Restrictions:**

* ❌ No LangChain, LlamaIndex, or agent frameworks  
* ❌ No managed vector DBs (Pinecone, etc.)  
* ✅ Build orchestration from scratch  
* ✅ Use pgvector

---

**Hints:**

* Start with single-service queries before multi-service  
* Prompt engineering for intent classification is critical  
* Metadata filtering \> pure vector search for speed  
* Google APIs fail often—implement retry with backoff  
* Store last 5 queries for conversation context ("that email")

---

## **Questions?**

Reach out for clarification. We value thoughtful architectural decisions over perfect implementations.

**Focus areas:** Orchestration logic, embedding quality, scalability design.

