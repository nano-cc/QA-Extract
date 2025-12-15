# QA Extract structure
- `clients/`: model clients and adapters (e.g., vLLM chat client).
- `pipeline/`: staged processing for ingest → chunking → QA generation → evaluation/refine → postprocess.
- `config/`: shared configuration and constants for the QA pipeline.

The directory layout follows the requirements in `QA-Extract/plan.md` for chunking, generation, judge/refine, and dedup steps.
