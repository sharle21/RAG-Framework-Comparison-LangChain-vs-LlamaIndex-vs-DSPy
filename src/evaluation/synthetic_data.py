"""
synthetic_data.py

Uses Ragas TestsetGenerator to generate ADDITIONAL adversarial-style QA pairs
on top of the RAGBench base dataset.

Purpose: RAGBench gives us clean, human-verified QA pairs for standard evaluation.
Ragas SDG gives us harder, more diverse questions for stress testing — particularly
multi-hop and reasoning questions that RAGBench doesn't cover well.

The two sets are kept separate:
  data/qa_pairs.json          ← RAGBench (standard benchmark, base truth)
  data/qa_pairs_synthetic.json ← Ragas SDG (harder questions, adversarial prep)

Run: python src/evaluation/synthetic_data.py
Costs: ~$0.10-0.20 in OpenAI API calls
"""


import json
import random
from pathlib import Path
from collections import defaultdict
from langchain_core.documents import Document as LCDocument
from dotenv import load_dotenv
from ragas.llms import LangchainLLMWrapper
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.testset.graph import KnowledgeGraph
from ragas.testset.persona import Persona
from ragas.testset.synthesizers import (
    SingleHopSpecificQuerySynthesizer,
    MultiHopAbstractQuerySynthesizer,
    MultiHopSpecificQuerySynthesizer
)

load_dotenv()

DATA_DIR = Path(__file__).parent.parent.parent / "data"
KG_PATH = DATA_DIR / "knowledge_graph.json"

def load_ragbench_as_lc_docs() -> list[LCDocument]:
    """Load RAGBench documents and inject noise for 2026 robustness testing."""
    docs_path = DATA_DIR / "raw" / "ragbench_documents.json"
    if not docs_path.exists():
        raise FileNotFoundError("RAGBench documents not found. Run prepare_data.py first.")

    with open(docs_path) as f:
        raw_docs = json.load(f)

    lc_docs = []
    for doc in raw_docs:
        content = doc.get("content", "")
        if len(content) < 100: continue
        lc_docs.append(LCDocument(
            page_content=content,
            metadata={
                "id": doc["id"],
                "title": doc["title"],
                "domain": doc.get("domain", "unknown")
            },
        ))

    print(f"Loaded {len(lc_docs)} documents")
    return lc_docs


def build_noise_docs() -> list[LCDocument]:
    """
    Domain-adjacent noise documents — look like they belong in the corpus
    but contain wrong or misleading information.

    These are injected into the corpus to test retrieval robustness.
    Unlike random noise (tortoise facts, cake recipes), domain-adjacent
    noise is a harder and more realistic test.
    """
    return [
        LCDocument(
            page_content="All financial derivatives are legally required to be settled in cash within 24 hours of issuance under international banking regulations.",
            metadata={"is_noise": True, "domain": "finqa"},
        ),
        LCDocument(
            page_content="IBM's technical support policy requires all enterprise tickets to be resolved within 2 hours regardless of severity level.",
            metadata={"is_noise": True, "domain": "techqa"},
        ),
        LCDocument(
            page_content="Studies have conclusively shown that COVID-19 vaccines provide no immunity benefit beyond 30 days of administration.",
            metadata={"is_noise": True, "domain": "covidqa"},
        ),
    ]



def generate_synthetic_qa(lc_docs: list[LCDocument], testset_size: int = 40) -> list[dict]:
    """
    Generate synthetic QA pairs using the Ragas v0.4.x API.

    GPT-4o builds the knowledge graph and handles single-hop questions.
    Mistral Large handles multi-hop questions — less refusal-heavy so it
    generates more creative and boundary-pushing questions.
    """
    
    # GPT-4o: knowledge graph + single-hop (precise, structured)
    gpt4o = LangchainLLMWrapper(
        ChatOpenAI(model="gpt-4o", temperature=0.8)
    )

    # Mistral Large: multi-hop (creative, less filtered)
    mistral = LangchainLLMWrapper(
        ChatMistralAI(model="mistral-large-latest", temperature=0.8)
    )

    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small")
    )

    # Adversarial persona — guides synthesizers to generate harder questions
    adversarial_persona = Persona(
        name="Adversarial Auditor",
        role_description=(
            "A skeptical auditor who tries to find contradictions in documents "
            "and asks trick questions with false premises to test system robustness."
        ),
    )

    if KG_PATH.exists():
        print(f"Loading existing knowledge graph from {KG_PATH}")
        kg = KnowledgeGraph.load(str(KG_PATH))
        print(f"  Loaded KG with {len(kg.nodes)} nodes — skipping rebuild")
        generator = TestsetGenerator(
            llm=gpt4o,
            embedding_model=embeddings,
            knowledge_graph=kg,
            persona_list=[adversarial_persona],
        )
    else:
        generator = TestsetGenerator(
            llm=gpt4o,                          # used for KG construction
            embedding_model=embeddings,
            persona_list=[adversarial_persona],
        )

    # Per-synthesizer model assignment:
    # GPT-4o for single-hop (factual precision matters more)
    # Mistral for multi-hop (creativity and less refusal matters more)
    query_distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=gpt4o), 0.4),
        (MultiHopAbstractQuerySynthesizer(llm=mistral), 0.3),
        (MultiHopSpecificQuerySynthesizer(llm=mistral), 0.3),
    ]

    print(f"\nGenerating {testset_size} synthetic QA pairs...")
    print("  Single-hop (GPT-4o):        40%")
    print("  Multi-hop abstract (Mistral): 30%")
    print("  Multi-hop specific (Mistral): 30%")
    print("  Persona: Adversarial Auditor")
    print("  Estimated cost: ~$0.20-0.30\n")

    dataset = generator.generate_with_langchain_docs(
        lc_docs,
        testset_size=testset_size,
        query_distribution=query_distribution,
        raise_exceptions=False,
    )

    if not KG_PATH.exists():
        generator.knowledge_graph.save(str(KG_PATH))
        print(f"  Saved knowledge graph to {KG_PATH}")

    df = dataset.to_pandas()

    qa_pairs = []
    for _, row in df.iterrows():
        question = row.get("user_input", row.get("question",""))
        answer =  row.get("reference", row.get("ground_truth",""))
        if not question or not answer: continue
        qa_pairs.append({
            "question": question,
            "ground_truth": answer,
            "question_type": str(row.get("synthesizer_name", "unknown")),
            "source": "ragas_synthetic"
        })

    print(f"Generated {len(qa_pairs)} QA pairs")
    by_type = defaultdict(int)
    for q in qa_pairs:
        by_type[q["question_type"]] += 1
    for t, count in by_type.items():
        print(f"  {t}: {count}")

    return qa_pairs

def main():
    all_docs = load_ragbench_as_lc_docs()

    # Balanced domain sampling to ensure variety
    by_domain = defaultdict(list)
    for doc in all_docs:
        by_domain[doc.metadata.get("domain", "unknown")].append(doc)

    sample_docs = []
    per_domain=30
    for domain, docs in by_domain.items():
        sample_docs.extend(docs[:per_domain]) # Take 30 per domain max
        print(f"  {domain}: {min(len(docs), per_domain)} docs")

    
    noise_docs = build_noise_docs()
    sample_docs.extend(noise_docs)
    print(f"  noise: {len(noise_docs)} injected documents")

    qa_pairs = generate_synthetic_qa(sample_docs, testset_size=40)

    out_path = DATA_DIR / "qa_pairs_synthetic.json"
    with open(out_path, "w") as f:
        json.dump(qa_pairs, f, indent=2)

    print(f"\nSaved to {out_path}")
    print(f"Knowledge graph saved to {KG_PATH}")
    print("These supplement qa_pairs.json (RAGBench) — used for adversarial attack generation.")

    print("\nSample questions:")
    for qa in qa_pairs[:3]:
        print(f"  [{qa['question_type']}] {qa['question'][:80]}...")

if __name__ == "__main__":
    main()