To run
1). Open Terminal as Command Prompt
2). Go to directory:  C:\Users\mante\Documents\Lonely_Octopus_AIAgent_Bootcamp\Legal_Logic_Analyzer>
3) Type:  python contract_gui_system.py 

AI LEGAL DOCUMENT ANALYZER
Description
  A multi-agent system for analyzing contract law appeals (motion to dismiss) and generating opposition analysis by identifying weaknesses in plaintiff's legal   arguments.
  Although this program was developed with the purpose of analyzing a Plaintiff's appeal for Motion to Dismiss it may be used for any legal document and          supporting case law. 
  ONE ARE OF OPTIMIZATION IS THE AI AGENT DESCRIPTIONS.  Admittedly, these have not been optimized.  
  (Note that the defendant submits a Motion to dismiss and the plaintiff appeal a Motion to Dismiss).

Overview
This system acts as opposing counsel to critically analyze Motion to Dismiss appeals in contract law cases. It uses specialized AI agents to examine procedural compliance, substantive law arguments, precedent analysis, and factual sufficiency to identify exploitable weaknesses.
System Architecture
Core Components

RAG Foundation: Vector database system for legal document storage and retrieval
Multi-Agent Analysis: Specialized AI agents for different aspects of legal analysis
GUI Interface: User-friendly interface for document selection and analysis
Opposition Report Generator: Professional brief output system

Agent Specialization

Procedural Analysis Agent (GPT-4o)

Legal standard compliance
Citation accuracy verification
Procedural rule adherence


Substantive Law Agent (Claude 3.5 Sonnet)

Contract formation analysis
Breach of contract evaluation
Damages theory assessment
Contract defenses identification


Precedent Analysis Agent (Claude 3.5 Sonnet)

Citation verification and characterization
Counter-authority identification
Factual distinction analysis


Factual Analysis Agent (GPT-4o)

Factual sufficiency evaluation
Internal consistency checking
Plausibility assessment


Lead Counsel Agent (Claude 3.5 Sonnet)

Strategic synthesis
Professional brief generation
Tactical recommendations



Requirements
API Keys

Pinecone API Key: Vector database storage
OpenAI API Key: GPT-4o agents and embeddings
Anthropic API Key: Claude 3.5 Sonnet agents

Python Dependencies
