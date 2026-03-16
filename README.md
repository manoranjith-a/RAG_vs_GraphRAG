ASRS Aviation Safety Query System
===================================
NASA ASRS dataset · 30,513 incidents · 2020-2025
Knowledge Graph + FAISS Semantic Search · GPT-4o-mini


WHAT THIS IS
------------
A local web application that queries the NASA Aviation Safety Reporting System
dataset using a GraphRAG pipeline. Ask any question in plain English and get:

  - A structured, cited answer from GPT-4o-mini
  - Top 5 semantically similar incidents retrieved via FAISS
  - A live knowledge graph traversal visualization (D3 force graph)
  - An intersection narrowing funnel showing how filters stack
  - Full query decomposition showing how your question mapped to graph nodes
  - Pattern frequency distributions across all matched incidents

The UI is split into two panels:
  LEFT  — RAG Pipeline (teammates connect their pipeline here)
  RIGHT — GraphRAG Pipeline (this repo)


FOLDER STRUCTURE
----------------
Your project folder should look like this:

  graphrag_app.py        <- Flask backend (this is what you run)
  graphrag_ui.html       <- Frontend (Flask serves this automatically)
  requirements.txt       <- Python dependencies
  README.txt             <- This file
  Dataset/
    asrs_graph.pkl       <- Knowledge graph (25.9 MB)
    faiss_index.bin      <- FAISS vector index (178.8 MB)
    faiss_metadata.pkl   <- Incident metadata (96.0 MB)

IMPORTANT: graphrag_app.py and graphrag_ui.html must be in the SAME folder.
           The Dataset/ folder must be directly inside that same folder.
           The CSV files are NOT needed to run the app.


ONE-TIME SETUP
--------------

1. Install dependencies

     pip install -r requirements.txt

   If you get an error on faiss, install the CPU version explicitly:

     pip install faiss-cpu

2. Add your OpenAI API key

   Open graphrag_app.py and find line 19:

     OPENAI_API_KEY = "your-api-key-here"

   Replace with your actual key:

     OPENAI_API_KEY = "sk-..."

   Each person needs their own key. Cost is roughly $0.0004 per query.
   Running all 15 evaluation queries costs about $0.006 total.


RUNNING THE APP
---------------

  1. Open a terminal in the folder containing graphrag_app.py

     cd /path/to/your/project/folder

  2. Run the server

     python graphrag_app.py

  3. Open your browser and go to

     http://localhost:5000

The app loads instantly. Do NOT open graphrag_ui.html directly in your browser
— it must be served by Flask to connect to the backend API.


USING THE APP
-------------

  - Type any aviation safety question in the query bar at the top
  - Press Enter or click ANALYZE
  - The RIGHT panel (GraphRAG) populates automatically
  - The LEFT panel (RAG) shows a placeholder until teammates connect their pipeline

What you see immediately (no scrolling needed):
  - Synthesized answer with direct answer, statistics, causal pathway,
    specific incident citations, and limitations
  - Top 5 retrieved incidents with ACN number, date, flight phase, outcome,
    and synopsis snippet

Scroll down in the GraphRAG panel to see:
  - Knowledge graph traversal — D3 force graph showing which nodes were
    traversed and how the intersection narrowed. Drag any node to explore.
  - Intersection narrowing funnel — bar chart showing how many incidents
    remained after each filter step (e.g. Night: 6,836 -> Fatigue: 178 -> Final Approach: 20)
  - Query decomposition — mode (single/intersect/compare/faiss_only) and
    colored pills showing anchor nodes, filter nodes, group comparisons, target types
  - Pattern distribution — frequency bars across all matched incidents,
    with side-by-side comparison layout for compare-mode queries


THE 15 EVALUATION QUERIES
--------------------------
Run all of these and record your answers for grading.

Category A — Counting & Summary (SQL pipeline expected to lead)
  1. How many incidents involved equipment failure as the primary problem?
  2. What are the most common anomaly types reported in the dataset?
  3. Summarize incidents involving bird strikes on commercial aircraft.

Category B — Cross-conditional (all pipelines competitive)
  4. What weather conditions are most commonly reported in runway incursion incidents?
  5. How do incidents during final approach differ from those during cruise phase?
  6. What corrective actions are most frequently taken after ATC-related incidents?

Category C — Causal & Multi-hop (GraphRAG expected to lead)
  7. When fatigue is a reported human factor, what other factors most commonly
     appear alongside it and what outcomes result?
  8. What chain of contributing factors most commonly precedes a near-miss or NMAC event?
  9. In night incidents, what combination of human factors and flight phase
     produces the most severe outcomes?
 10. How does the pattern of contributing factors differ between incidents where
     passengers were involved versus those where they were not?
 11. What sequence of events most commonly leads to a go-around or missed approach?
 12. When communication breakdown is a factor, which other human factors are
     almost always also present?
 13. What distinguishes incidents that resulted in aircraft damage from those
     with no action taken?
 14. Across incidents involving Boeing 737s, what are the most common causal
     pathways leading to procedural deviations?
 15. What systemic factors appear repeatedly across incidents where the crew
     had to declare an emergency?


FOR TEAMMATES — CONNECTING THE RAG PIPELINE
--------------------------------------------
The LEFT panel is ready for your RAG pipeline output. To connect it:

  1. In graphrag_ui.html, find the function renderRAGPlaceholder()
     around line 460 in the <script> section.

  2. Replace that function with a call to your own RAG endpoint, or
     modify the submit() function to call both APIs in parallel.

  3. The RAG panel expects the same data shape as the GraphRAG response:
     {
       "answer":    string,
       "incidents": [ { acn, date, flight_phase, primary_problem, result, synopsis,
                        similarity_score }, ... ],
       "cost_usd":  number,
       "tokens":    { prompt: number, output: number },
       "time_seconds": number
     }

  4. Use the renderIncs() function already in the HTML to render your
     retrieved incidents in the same format as the GraphRAG side.


TROUBLESHOOTING
---------------

"Address already in use" — something else is on port 5000. Kill it:

  Mac/Linux:
    lsof -ti:5000 | xargs kill -9

  Windows:
    netstat -ano | findstr :5000
    taskkill /PID <PID> /F

"No module named faiss":
    pip install faiss-cpu

"FileNotFoundError: Dataset/..." — you are not running from the right folder:
    cd /path/to/folder/containing/graphrag_app.py
    python graphrag_app.py

"Graph not rendering" — make sure you access via http://localhost:5000,
not by opening the HTML file directly from your file system.

"API key error" — check that your key starts with sk- and has no extra spaces.
