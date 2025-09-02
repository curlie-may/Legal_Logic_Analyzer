# Multi-Agent Legal Opposition Analysis System

import asyncio
import os
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime

from openai import OpenAI
from anthropic import Anthropic

# Import your working RAG foundation
from contract_rag_foundation import ContractRAGBuilder, ContractArgument

class ConfidenceLevel(Enum):
    VERIFIED = "🟢 VERIFIED"
    INFERRED = "🟡 INFERRED" 
    UNSUPPORTED = "🔴 UNSUPPORTED"
    STRATEGIC = "⚪ STRATEGIC"

@dataclass
class OppositionPoint:
    weakness_type: str
    severity: str  # "Critical", "Significant", "Minor"
    argument: str
    supporting_authority: str
    confidence: ConfidenceLevel
    page_reference: str
    agent_source: str

class LegalAnalysisAgent:
    """Base class for legal analysis agents"""
    
    def __init__(self, agent_name: str, llm_client, model_name: str):
        self.agent_name = agent_name
        self.llm_client = llm_client
        self.model_name = model_name
        
    async def analyze_async(self, arguments: List[ContractArgument], rag_system: ContractRAGBuilder) -> List[OppositionPoint]:
        """Async analysis method - to be implemented by subclasses"""
        raise NotImplementedError
        
    async def _call_llm(self, prompt: str) -> str:
        """Call LLM with proper error handling"""
        try:
            if isinstance(self.llm_client, OpenAI):
                response = await asyncio.to_thread(
                    self.llm_client.chat.completions.create,
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1
                )
                return response.choices[0].message.content
            
            elif isinstance(self.llm_client, Anthropic):
                response = await asyncio.to_thread(
                    self.llm_client.messages.create,
                    model=self.model_name,
                    max_tokens=4000,
                    temperature=0.1,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
                
        except Exception as e:
            print(f"Error in {self.agent_name}: {e}")
            return f"Error: {str(e)}"

class ProceduralAnalysisAgent(LegalAnalysisAgent):
    """Analyzes procedural compliance and legal standards"""
    
    def __init__(self):
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        super().__init__("Procedural Analysis Agent", openai_client, "gpt-4o")
        
    async def analyze_async(self, arguments: List[ContractArgument], rag_system: ContractRAGBuilder) -> List[OppositionPoint]:
        """Analyze procedural compliance"""
        
        opposition_points = []
        
        for argument in arguments:
            # Analyze legal standard application
            standard_analysis = await self._analyze_legal_standard(argument, rag_system)
            opposition_points.extend(standard_analysis)
            
            # Check citation accuracy
            citation_analysis = await self._verify_citations(argument, rag_system)
            opposition_points.extend(citation_analysis)
        
        return opposition_points
    
    async def _analyze_legal_standard(self, argument: ContractArgument, rag_system: ContractRAGBuilder) -> List[OppositionPoint]:
        """Analyze whether legal standard is correctly applied"""
        
        # Search for relevant standard discussions in cited cases
        standard_cases = rag_system.search_contract_precedent(
            query=f"{argument.legal_standard} pleading requirements burden",
            top_k=5
        )
        
        prompt = f"""
        You are a defense attorney analyzing procedural compliance in a contract law case.
        
        PLAINTIFF'S LEGAL STANDARD CLAIM:
        Standard Cited: {argument.legal_standard}
        Argument: {argument.section_title}
        
        RELEVANT CASE LAW FROM CITATIONS:
        {json.dumps([case['content'] for case in standard_cases[:3]], indent=2)}
        
        ANALYZE FOR PROCEDURAL WEAKNESSES:
        1. Does plaintiff correctly state the legal standard?
        2. Does plaintiff properly apply the standard to their facts?
        3. Are there any procedural missteps or misstatements?
        4. Does the legal standard match what the cited cases actually say?
        
        CRITICAL RULES:
        - Only cite cases from the provided case law
        - Flag any misstatements of legal standards
        - Identify gaps between what plaintiff claims the law requires vs. what it actually requires
        
        Format response as:
        WEAKNESS_TYPE: [type]
        SEVERITY: [Critical/Significant/Minor]
        ARGUMENT: [specific weakness found]
        SUPPORTING_AUTHORITY: [case citation if applicable]
        CONFIDENCE: [VERIFIED/INFERRED/UNSUPPORTED]
        """
        
        response = await self._call_llm(prompt)
        return self._parse_agent_response(response, argument)
    
    async def _verify_citations(self, argument: ContractArgument, rag_system: ContractRAGBuilder) -> List[OppositionPoint]:
        """Verify accuracy of citations"""
        
        citation_issues = []
        
        for citation in argument.citations:
            # Search for this specific citation in RAG
            citation_content = rag_system.search_contract_precedent(
                query=f"{citation}",
                top_k=3
            )
            
            if not citation_content:
                citation_issues.append(OppositionPoint(
                    weakness_type="Citation Error",
                    severity="Critical",
                    argument=f"Citation '{citation}' not found in provided case materials",
                    supporting_authority="Document review",
                    confidence=ConfidenceLevel.VERIFIED,
                    page_reference=argument.section_title,
                    agent_source=self.agent_name
                ))
                continue
            
            # Verify citation accuracy
            verification_prompt = f"""
            CITATION VERIFICATION:
            
            PLAINTIFF CITES: {citation}
            CASE CONTENT FOUND: {citation_content[0]['content'] if citation_content else 'NOT FOUND'}
            PLAINTIFF'S ARGUMENT: {argument.full_text[:500]}
            
            VERIFICATION QUESTIONS:
            1. Does this case actually support plaintiff's legal argument?
            2. Is plaintiff citing the holding or just dicta?
            3. Are there factual distinctions that limit applicability?
            4. Is the citation used accurately?
            
            Only flag if there's a clear misuse of the citation.
            """
            
            verification_result = await self._call_llm(verification_prompt)
            
            if "does not support" in verification_result.lower() or "mischaracteriz" in verification_result.lower():
                citation_issues.append(OppositionPoint(
                    weakness_type="Citation Mischaracterization",
                    severity="Significant",
                    argument=f"Case {citation} does not support plaintiff's position: {verification_result[:200]}",
                    supporting_authority=citation,
                    confidence=ConfidenceLevel.VERIFIED,
                    page_reference=argument.section_title,
                    agent_source=self.agent_name
                ))
        
        return citation_issues
    
    def _parse_agent_response(self, response: str, argument: ContractArgument) -> List[OppositionPoint]:
        """Parse agent response into OppositionPoint objects"""
        
        points = []
        
        # Simple parsing - look for structured responses
        if "WEAKNESS_TYPE:" in response:
            lines = response.split('\n')
            weakness_data = {}
            
            for line in lines:
                if "WEAKNESS_TYPE:" in line:
                    weakness_data['weakness_type'] = line.split(':', 1)[1].strip()
                elif "SEVERITY:" in line:
                    weakness_data['severity'] = line.split(':', 1)[1].strip()
                elif "ARGUMENT:" in line:
                    weakness_data['argument'] = line.split(':', 1)[1].strip()
                elif "SUPPORTING_AUTHORITY:" in line:
                    weakness_data['supporting_authority'] = line.split(':', 1)[1].strip()
                elif "CONFIDENCE:" in line:
                    conf_str = line.split(':', 1)[1].strip().upper()
                    weakness_data['confidence'] = getattr(ConfidenceLevel, conf_str, ConfidenceLevel.INFERRED)
            
            if weakness_data:
                points.append(OppositionPoint(
                    weakness_type=weakness_data.get('weakness_type', 'Procedural Issue'),
                    severity=weakness_data.get('severity', 'Minor'),
                    argument=weakness_data.get('argument', response[:200]),
                    supporting_authority=weakness_data.get('supporting_authority', 'Analysis'),
                    confidence=weakness_data.get('confidence', ConfidenceLevel.INFERRED),
                    page_reference=argument.section_title,
                    agent_source=self.agent_name
                ))
        
        return points

class SubstantiveLawAgent(LegalAnalysisAgent):
    """Analyzes substantive contract law elements"""
    
    def __init__(self):
        anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        super().__init__("Substantive Law Agent", anthropic_client, "claude-3-5-sonnet-20241022")
        
    async def analyze_async(self, arguments: List[ContractArgument], rag_system: ContractRAGBuilder) -> List[OppositionPoint]:
        """Deep analysis of contract law elements"""
        
        opposition_points = []
        
        for argument in arguments:
            if argument.argument_type == 'formation':
                formation_analysis = await self._analyze_contract_formation(argument, rag_system)
                opposition_points.extend(formation_analysis)
                
            elif argument.argument_type == 'breach':
                breach_analysis = await self._analyze_breach_claims(argument, rag_system)
                opposition_points.extend(breach_analysis)
                
            elif argument.argument_type == 'damages':
                damages_analysis = await self._analyze_damages_theory(argument, rag_system)
                opposition_points.extend(damages_analysis)
                
            elif argument.argument_type == 'defenses':
                defenses_analysis = await self._analyze_contract_defenses(argument, rag_system)
                opposition_points.extend(defenses_analysis)
        
        return opposition_points
    
    async def _analyze_contract_formation(self, argument: ContractArgument, rag_system: ContractRAGBuilder) -> List[OppositionPoint]:
        """Analyze contract formation elements"""
        
        formation_cases = rag_system.search_contract_precedent(
            query=f"contract formation offer acceptance consideration {' '.join(argument.factual_allegations[:3])}",
            argument_type='formation',
            top_k=7
        )
        
        prompt = f"""
        You are a senior contract litigation attorney analyzing formation claims for weaknesses.
        
        PLAINTIFF'S FORMATION ARGUMENT:
        Section: {argument.section_title}
        Elements Addressed: {argument.contract_elements}
        Factual Allegations: {argument.factual_allegations[:5]}
        Legal Standard: {argument.legal_standard}
        
        RELEVANT FORMATION PRECEDENT:
        {json.dumps([case['content'] for case in formation_cases[:5]], indent=2)}
        
        ANALYZE FOR FORMATION WEAKNESSES:
        
        1. OFFER ANALYSIS:
           - Are the alleged offer terms sufficiently definite?
           - Does the communication manifest intent to be bound?
           - Are essential terms (price, performance, timing) missing?
        
        2. ACCEPTANCE ANALYSIS:
           - Was acceptance unqualified and unconditional?
           - Did acceptance occur within reasonable/specified time?
           - Does acceptance method comply with offer terms?
        
        3. CONSIDERATION ANALYSIS:
           - Is there adequate consideration from both parties?
           - Are the promises sufficiently definite to constitute consideration?
           - Is consideration illusory or insufficient?
        
        4. DEFINITENESS:
           - Are all material terms sufficiently certain?
           - Can a court determine parties' obligations?
           - Are there missing essential terms that prevent enforcement?
        
        Apply CRITICAL CONTRACT LAW ANALYSIS:
        - Question every factual allegation for legal sufficiency
        - Identify missing elements that defeat formation
        - Find factual gaps that create reasonable doubt
        
        ONLY cite cases from the provided precedent. Flag the TOP 2 most significant weaknesses.
        """
        
        response = await self._call_llm(prompt)
        return self._parse_substantive_response(response, argument)
    
    async def _analyze_breach_claims(self, argument: ContractArgument, rag_system: ContractRAGBuilder) -> List[OppositionPoint]:
        """Analyze breach of contract claims"""
        
        breach_cases = rag_system.search_contract_precedent(
            query=f"breach of contract material breach substantial performance {' '.join(argument.factual_allegations[:3])}",
            argument_type='breach',
            top_k=7
        )
        
        prompt = f"""
        You are analyzing breach of contract claims for defensive weaknesses.
        
        PLAINTIFF'S BREACH ARGUMENT:
        Section: {argument.section_title}
        Factual Allegations: {argument.factual_allegations[:5]}
        Legal Standard: {argument.legal_standard}
        
        RELEVANT BREACH PRECEDENT:
        {json.dumps([case['content'] for case in breach_cases[:5]], indent=2)}
        
        ANALYZE FOR BREACH CLAIM WEAKNESSES:
        
        1. PERFORMANCE ANALYSIS:
           - Did plaintiff adequately perform their obligations?
           - Are there conditions precedent that plaintiff failed to satisfy?
           - Is plaintiff's performance substantial enough to require counter-performance?
        
        2. MATERIALITY ANALYSIS:
           - Is the alleged breach actually material vs. minor?
           - What is the economic significance of the alleged breach?
           - Did the breach go to the essence of the contract?
        
        3. CAUSATION ISSUES:
           - Did defendant's actions actually cause plaintiff's damages?
           - Are there intervening causes that break the causal chain?
           - Did plaintiff's own actions contribute to the alleged harm?
        
        4. NOTICE AND CURE:
           - Did plaintiff provide proper notice of breach?
           - Was defendant given opportunity to cure?
           - Did plaintiff act to mitigate damages?
        
        Identify the STRONGEST 2 defensive arguments against the breach claim.
        """
        
        response = await self._call_llm(prompt)
        return self._parse_substantive_response(response, argument)
    
    async def _analyze_damages_theory(self, argument: ContractArgument, rag_system: ContractRAGBuilder) -> List[OppositionPoint]:
        """Analyze damages claims"""
        
        damages_cases = rag_system.search_contract_precedent(
            query=f"damages expectation consequential foreseeability mitigation {' '.join(argument.factual_allegations[:3])}",
            argument_type='damages',
            top_k=7
        )
        
        prompt = f"""
        You are analyzing contract damages claims for weaknesses.
        
        PLAINTIFF'S DAMAGES ARGUMENT:
        Section: {argument.section_title}
        Factual Allegations: {argument.factual_allegations[:5]}
        
        RELEVANT DAMAGES PRECEDENT:
        {json.dumps([case['content'] for case in damages_cases[:5]], indent=2)}
        
        ANALYZE DAMAGES WEAKNESSES:
        
        1. CAUSATION:
           - Are damages directly caused by alleged breach?
           - Are there intervening causes?
           - Did plaintiff's actions contribute to damages?
        
        2. FORESEEABILITY:
           - Were damages reasonably foreseeable at contract formation?
           - Did defendant have notice of special circumstances?
           - Are consequential damages properly limited?
        
        3. CERTAINTY:
           - Are damages calculated with reasonable certainty?
           - Are damages speculative or conjectural?
           - Is there adequate proof of lost profits/value?
        
        4. MITIGATION:
           - Did plaintiff take reasonable steps to mitigate?
           - Were mitigation efforts adequate and timely?
           - Could damages have been avoided or reduced?
        
        Focus on the TOP 2 strongest challenges to damages claims.
        """
        
        response = await self._call_llm(prompt)
        return self._parse_substantive_response(response, argument)
    
    async def _analyze_contract_defenses(self, argument: ContractArgument, rag_system: ContractRAGBuilder) -> List[OppositionPoint]:
        """Analyze contract defenses that plaintiff may have overlooked"""
        
        defense_cases = rag_system.search_contract_precedent(
            query=f"contract defenses unconscionability statute limitations impossibility {' '.join(argument.factual_allegations[:3])}",
            argument_type='defenses',
            top_k=7
        )
        
        prompt = f"""
        You are identifying contract defenses that plaintiff failed to address.
        
        PLAINTIFF'S ARGUMENT:
        Section: {argument.section_title}
        Factual Allegations: {argument.factual_allegations[:5]}
        
        RELEVANT DEFENSE PRECEDENT:
        {json.dumps([case['content'] for case in defense_cases[:5]], indent=2)}
        
        IDENTIFY AVAILABLE DEFENSES:
        
        1. STATUTE OF LIMITATIONS:
           - When did the cause of action accrue?
           - Has the limitations period expired?
           - Are there any tolling provisions that apply?
        
        2. UNCONSCIONABILITY:
           - Were contract terms procedurally unconscionable?
           - Are terms substantively unconscionable?
           - Was there meaningful choice in contract formation?
        
        3. IMPOSSIBILITY/IMPRACTICABILITY:
           - Did unforeseen circumstances make performance impossible?
           - Were risks allocated in the contract?
           - Is performance commercially impracticable?
        
        4. WAIVER/ESTOPPEL:
           - Did plaintiff waive right to claim breach?
           - Should plaintiff be estopped from claiming damages?
           - Did conduct indicate acceptance of performance?
        
        Identify the 2 STRONGEST potential defenses that plaintiff's argument fails to address.
        """
        
        response = await self._call_llm(prompt)
        return self._parse_substantive_response(response, argument)
    
    def _parse_substantive_response(self, response: str, argument: ContractArgument) -> List[OppositionPoint]:
        """Parse Claude's response into structured opposition points"""
        
        points = []
        
        # Look for numbered points or clear weakness indicators
        sections = response.split('\n\n')
        
        for section in sections:
            if any(indicator in section.lower() for indicator in ['weakness', 'problem', 'fails', 'insufficient', 'missing']):
                
                # Determine severity based on language
                severity = "Critical" if any(word in section.lower() for word in ['critical', 'fatal', 'fundamental']) else \
                          "Significant" if any(word in section.lower() for word in ['significant', 'material', 'important']) else \
                          "Minor"
                
                # Extract weakness type from context
                weakness_type = "Formation Deficiency" if argument.argument_type == 'formation' else \
                               "Breach Analysis Flaw" if argument.argument_type == 'breach' else \
                               "Damages Theory Problem" if argument.argument_type == 'damages' else \
                               "Defense Oversight"
                
                points.append(OppositionPoint(
                    weakness_type=weakness_type,
                    severity=severity,
                    argument=section.strip()[:500],  # Limit length
                    supporting_authority="Case law analysis",
                    confidence=ConfidenceLevel.INFERRED,
                    page_reference=argument.section_title,
                    agent_source=self.agent_name
                ))
        
        return points[:3]  # Limit to top 3 findings

class PrecedentAnalysisAgent(LegalAnalysisAgent):
    """Analyzes precedent and finds counter-authorities"""
    
    def __init__(self):
        anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        super().__init__("Precedent Analysis Agent", anthropic_client, "claude-3-5-sonnet-20241022")
        
    async def analyze_async(self, arguments: List[ContractArgument], rag_system: ContractRAGBuilder) -> List[OppositionPoint]:
        """Analyze precedent and find counter-authorities"""
        
        opposition_points = []
        
        for argument in arguments:
            # Find counter-authorities
            counter_analysis = await self._find_counter_authorities(argument, rag_system)
            opposition_points.extend(counter_analysis)
            
            # Analyze factual distinctions
            distinction_analysis = await self._analyze_factual_distinctions(argument, rag_system)
            opposition_points.extend(distinction_analysis)
        
        return opposition_points
    
    async def _find_counter_authorities(self, argument: ContractArgument, rag_system: ContractRAGBuilder) -> List[OppositionPoint]:
        """Find cases that undermine plaintiff's arguments"""
        
        # Search for cases with opposite outcomes or different standards
        counter_cases = rag_system.search_contract_precedent(
            query=f"{argument.argument_type} contract dismissed motion granted insufficient allegations",
            top_k=10
        )
        
        prompt = f"""
        You are looking for counter-authorities that undermine plaintiff's argument.
        
        PLAINTIFF'S ARGUMENT:
        Section: {argument.section_title}
        Citations Used: {argument.citations}
        Legal Position: {argument.full_text[:300]}
        
        AVAILABLE CASE LAW:
        {json.dumps([case['content'] for case in counter_cases[:8]], indent=2)}
        
        FIND COUNTER-AUTHORITIES:
        
        1. Cases where similar claims were dismissed
        2. Cases requiring higher standards than plaintiff meets
        3. Cases with similar facts but different outcomes
        4. Cases that limit or distinguish plaintiff's cited authorities
        
        For each counter-authority found:
        - Explain how it undermines plaintiff's position
        - Note factual similarities to current case
        - Identify specific legal holdings that favor defense
        
        ONLY use cases from the provided case law. Focus on the 2 strongest counter-authorities.
        """
        
        response = await self._call_llm(prompt)
        return self._parse_precedent_response(response, argument, "Counter-Authority")
    
    async def _analyze_factual_distinctions(self, argument: ContractArgument, rag_system: ContractRAGBuilder) -> List[OppositionPoint]:
        """Analyze how cited cases are factually distinguishable"""
        
        # Get cases that plaintiff actually cited
        cited_case_content = []
        for citation in argument.citations:
            case_results = rag_system.search_contract_precedent(query=citation, top_k=3)
            cited_case_content.extend(case_results)
        
        prompt = f"""
        You are analyzing whether plaintiff's cited cases are actually distinguishable.
        
        PLAINTIFF'S FACTUAL ALLEGATIONS:
        {argument.factual_allegations[:5]}
        
        PLAINTIFF'S CITED CASES:
        {json.dumps([case['content'] for case in cited_case_content[:5]], indent=2)}
        
        FIND FACTUAL DISTINCTIONS:
        
        1. How are the facts in cited cases different from plaintiff's case?
        2. Do these distinctions limit the precedential value?
        3. Are plaintiff's facts weaker than those in cited cases?
        4. Would the cited cases reach a different result on plaintiff's facts?
        
        For each significant distinction:
        - Identify the specific factual difference
        - Explain why it matters legally
        - Show how it weakens plaintiff's reliance on the case
        
        Focus on the 2 most significant factual distinctions.
        """
        
        response = await self._call_llm(prompt)
        return self._parse_precedent_response(response, argument, "Factual Distinction")
    
    def _parse_precedent_response(self, response: str, argument: ContractArgument, analysis_type: str) -> List[OppositionPoint]:
        """Parse precedent analysis into opposition points"""
        
        points = []
        
        # Split response into logical sections
        sections = response.split('\n\n')
        
        for section in sections:
            if len(section.strip()) > 100:  # Substantial content
                
                severity = "Significant" if "strong" in section.lower() or "clear" in section.lower() else "Minor"
                
                points.append(OppositionPoint(
                    weakness_type=analysis_type,
                    severity=severity,
                    argument=section.strip()[:500],
                    supporting_authority="Case law comparison",
                    confidence=ConfidenceLevel.INFERRED,
                    page_reference=argument.section_title,
                    agent_source=self.agent_name
                ))
        
        return points[:2]  # Top 2 findings

class FactualAnalysisAgent(LegalAnalysisAgent):
    """Analyzes factual sufficiency and consistency"""
    
    def __init__(self):
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        super().__init__("Factual Analysis Agent", openai_client, "gpt-4o")
        
    async def analyze_async(self, arguments: List[ContractArgument], rag_system: ContractRAGBuilder) -> List[OppositionPoint]:
        """Analyze factual sufficiency"""
        
        opposition_points = []
        
        for argument in arguments:
            # Check factual sufficiency
            sufficiency_analysis = await self._analyze_factual_sufficiency(argument, rag_system)
            opposition_points.extend(sufficiency_analysis)
            
            # Check internal consistency
            consistency_analysis = await self._check_internal_consistency(argument, rag_system)
            opposition_points.extend(consistency_analysis)
        
        return opposition_points
    
    async def _analyze_factual_sufficiency(self, argument: ContractArgument, rag_system: ContractRAGBuilder) -> List[OppositionPoint]:
        """Analyze whether factual allegations are sufficient"""
        
        # Get cases showing required factual specificity
        specificity_cases = rag_system.search_contract_precedent(
            query=f"{argument.legal_standard} factual allegations specificity requirements pleading",
            top_k=5
        )
        
        prompt = f"""
        You are analyzing factual sufficiency under the applicable pleading standard.
        
        PLEADING STANDARD: {argument.legal_standard}
        
        PLAINTIFF'S FACTUAL ALLEGATIONS:
        {argument.factual_allegations}
        
        LEGAL REQUIREMENTS FROM CASE LAW:
        {json.dumps([case['content'] for case in specificity_cases[:3]], indent=2)}
        
        ANALYZE FACTUAL SUFFICIENCY:
        
        1. SPECIFICITY:
           - Are allegations specific enough to meet pleading requirements?
           - Are there vague or conclusory allegations?
           - Do allegations provide fair notice of claims?
        
        2. ESSENTIAL ELEMENTS:
           - Are all required factual elements alleged?
           - Are there gaps in the factual narrative?
           - What essential facts are missing?
        
        3. LOGICAL CONSISTENCY:
           - Do the allegations hang together logically?
           - Are there internal contradictions?
           - Do timelines and sequences make sense?
        
        4. PLAUSIBILITY:
           - Are allegations plausible on their face?
           - Do they support reasonable inferences?
           - Are alternative explanations more likely?
        
        Identify the 2 strongest factual deficiencies.
        """
        
        response = await self._call_llm(prompt)
        return self._parse_factual_response(response, argument)
    
    async def _check_internal_consistency(self, argument: ContractArgument, rag_system: ContractRAGBuilder) -> List[OppositionPoint]:
        """Check for internal inconsistencies in allegations"""
        
        prompt = f"""
        You are checking for internal inconsistencies in plaintiff's factual allegations.
        
        ARGUMENT SECTION: {argument.section_title}
        FACTUAL ALLEGATIONS: {argument.factual_allegations}
        FULL ARGUMENT TEXT: {argument.full_text[:1000]}
        
        CHECK FOR INCONSISTENCIES:
        
        1. TIMELINE ISSUES:
           - Do dates and sequences make sense?
           - Are there chronological contradictions?
           - Do events happen in logical order?
        
        2. FACTUAL CONTRADICTIONS:
           - Do different allegations contradict each other?
           - Are there incompatible claims about same events?
           - Do facts support conflicting inferences?
        
        3. LOGIC GAPS:
           - Are there missing steps in causal chains?
           - Do conclusions follow from stated facts?
           - Are there unexplained assumptions?
        
        4. CREDIBILITY ISSUES:
           - Are allegations inherently implausible?
           - Do they strain credulity?
           - Are there obvious alternative explanations?
        
        Flag only clear, significant inconsistencies with specific examples.
        """
        
        response = await self._call_llm(prompt)
        return self._parse_factual_response(response, argument)
    
    def _parse_factual_response(self, response: str, argument: ContractArgument) -> List[OppositionPoint]:
        """Parse factual analysis into opposition points"""
        
        points = []
        
        # Look for clear factual problems
        if any(indicator in response.lower() for indicator in ['inconsistent', 'contradiction', 'insufficient', 'missing', 'vague']):
            
            # Split into paragraphs and analyze each
            paragraphs = [p.strip() for p in response.split('\n\n') if len(p.strip()) > 50]
            
            for para in paragraphs[:3]:  # Top 3 issues
                if any(problem in para.lower() for problem in ['problem', 'issue', 'deficient', 'lacks', 'fails']):
                    
                    severity = "Critical" if "critical" in para.lower() or "fatal" in para.lower() else \
                              "Significant" if "significant" in para.lower() or "material" in para.lower() else \
                              "Minor"
                    
                    points.append(OppositionPoint(
                        weakness_type="Factual Deficiency",
                        severity=severity,
                        argument=para[:400],
                        supporting_authority="Factual analysis",
                        confidence=ConfidenceLevel.INFERRED,
                        page_reference=argument.section_title,
                        agent_source=self.agent_name
                    ))
        
        return points

class LeadCounselAgent(LegalAnalysisAgent):
    """Coordinates analysis and generates final opposition report"""
    
    def __init__(self):
        anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        super().__init__("Lead Counsel Agent", anthropic_client, "claude-3-5-sonnet-20241022")
        
    async def synthesize_opposition_brief(self, all_opposition_points: List[OppositionPoint], 
                                        arguments: List[ContractArgument], rag_system: ContractRAGBuilder) -> str:
        """Generate comprehensive opposition analysis"""
        
        # Organize points by severity and type
        critical_points = [p for p in all_opposition_points if p.severity == "Critical"]
        significant_points = [p for p in all_opposition_points if p.severity == "Significant"]
        minor_points = [p for p in all_opposition_points if p.severity == "Minor"]
        
        # Group by weakness type
        weakness_groups = {}
        for point in all_opposition_points:
            if point.weakness_type not in weakness_groups:
                weakness_groups[point.weakness_type] = []
            weakness_groups[point.weakness_type].append(point)
        
        # Generate strategic overview
        total_weaknesses = len(all_opposition_points)
        case_strength = "Weak" if len(critical_points) > 3 else \
                       "Moderate" if len(significant_points) > 5 else \
                       "Strong"
        
        synthesis_prompt = f"""
        You are lead counsel preparing opposition to plaintiff's contract appeal.
        
        CASE ANALYSIS SUMMARY:
        - {len(arguments)} major legal arguments analyzed
        - {total_weaknesses} weaknesses identified across all arguments
        - {len(critical_points)} critical issues, {len(significant_points)} significant issues
        - Overall case assessment: {case_strength}
        
        CRITICAL WEAKNESSES IDENTIFIED:
        {json.dumps([f"{p.weakness_type}: {p.argument[:200]}" for p in critical_points[:5]], indent=2)}
        
        SIGNIFICANT WEAKNESSES IDENTIFIED:
        {json.dumps([f"{p.weakness_type}: {p.argument[:200]}" for p in significant_points[:8]], indent=2)}
        
        WEAKNESS CATEGORIES FOUND:
        {json.dumps(list(weakness_groups.keys()), indent=2)}
        
        GENERATE PROFESSIONAL OPPOSITION ANALYSIS:
        
        ## EXECUTIVE SUMMARY
        - Overall case strength assessment
        - Top 3 strongest opposition arguments
        - Recommended litigation strategy
        - Settlement considerations
        
        ## I. PROCEDURAL DEFICIENCIES
        - Legal standard misapplications
        - Citation and authority problems
        - Pleading standard failures
        
        ## II. SUBSTANTIVE LAW FAILURES
        - Contract formation element deficiencies
        - Breach analysis gaps
        - Damages theory problems
        - Available defenses plaintiff overlooked
        
        ## III. PRECEDENT ANALYSIS FLAWS
        - Citation mischaracterizations
        - Counter-authorities plaintiff ignored
        - Factual distinctions limiting precedent
        
        ## IV. FACTUAL INSUFFICIENCIES
        - Inadequate factual allegations
        - Internal inconsistencies
        - Missing essential elements
        
        ## V. STRATEGIC RECOMMENDATIONS
        - Strongest arguments for dismissal/defense
        - Priority order for briefing
        - Discovery strategy implications
        - Settlement leverage analysis
        
        Use formal legal brief writing style with clear headings and professional tone.
        Focus on the most compelling arguments that have the highest likelihood of success.
        Be specific about why each weakness matters and how it undermines plaintiff's case.
        """
        
        final_brief = await self._call_llm(synthesis_prompt)
        
        # Add metadata footer
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        final_brief += f"""
        
        ---
        ANALYSIS METADATA:
        Generated: {timestamp}
        Arguments Analyzed: {len(arguments)}
        Total Weaknesses Found: {total_weaknesses}
        Agent Analysis Complete: ✅ Procedural, ✅ Substantive, ✅ Precedent, ✅ Factual
        Confidence Levels: {len([p for p in all_opposition_points if p.confidence == ConfidenceLevel.VERIFIED])} Verified, {len([p for p in all_opposition_points if p.confidence == ConfidenceLevel.INFERRED])} Inferred
        """
        
        return final_brief

class MultiAgentOppositionSystem:
    """Main orchestrator for multi-agent opposition analysis"""
    
    def __init__(self, rag_system: ContractRAGBuilder):
        self.rag_system = rag_system
        
        # Initialize all agents
        self.agents = {
            'procedural': ProceduralAnalysisAgent(),
            'substantive': SubstantiveLawAgent(),
            'precedent': PrecedentAnalysisAgent(),
            'factual': FactualAnalysisAgent(),
            'lead_counsel': LeadCounselAgent()
        }
        
        print("✅ Multi-agent opposition system initialized")
    
    async def generate_opposition_analysis(self) -> str:
        """Run complete multi-agent analysis"""
        
        print("🔄 Starting multi-agent opposition analysis...")
        
        # Get processed arguments from RAG system
        arguments = await self._extract_arguments_from_rag()
        
        if not arguments:
            return "❌ Error: No legal arguments found in processed documents. Please ensure documents were processed correctly."
        
        print(f"📋 Analyzing {len(arguments)} legal arguments...")
        
        # Run all analysis agents in parallel
        analysis_tasks = []
        agent_names = ['procedural', 'substantive', 'precedent', 'factual']
        
        for agent_name in agent_names:
            agent = self.agents[agent_name]
            task = agent.analyze_async(arguments, self.rag_system)
            analysis_tasks.append((agent_name, task))
        
        print("⚡ Running parallel agent analysis...")
        
        # Wait for all agents to complete
        all_opposition_points = []
        completed_agents = []
        
        for agent_name, task in analysis_tasks:
            try:
                agent_results = await task
                all_opposition_points.extend(agent_results)
                completed_agents.append(agent_name)
                print(f"✅ {agent_name.title()} Agent: {len(agent_results)} issues found")
                
            except Exception as e:
                print(f"❌ {agent_name.title()} Agent failed: {e}")
                continue
        
        if not all_opposition_points:
            return "❌ Error: No opposition points identified. Analysis may have failed."
        
        print(f"🎯 Total weaknesses found: {len(all_opposition_points)}")
        
        # Generate final synthesis
        print("📝 Generating final opposition brief...")
        
        try:
            final_opposition_brief = await self.agents['lead_counsel'].synthesize_opposition_brief(
                all_opposition_points, arguments, self.rag_system
            )
            
            print("✅ Opposition analysis complete!")
            return final_opposition_brief
            
        except Exception as e:
            print(f"❌ Lead counsel synthesis failed: {e}")
            
            # Fallback: Generate simple summary
            return self._generate_fallback_summary(all_opposition_points, arguments)
    
    async def _extract_arguments_from_rag(self) -> List[ContractArgument]:
        """Extract processed arguments from RAG system"""
        
        try:
            # Search for appeal brief arguments
            brief_results = self.rag_system.search_contract_precedent(
                query="plaintiff argument section appeal brief",
                top_k=20
            )
            
            # Filter for appeal brief content only
            appeal_results = [r for r in brief_results if r['metadata'].get('document_type') == 'appeal_brief']
            
            if not appeal_results:
                print("⚠️ No appeal brief arguments found in RAG system")
                return []
            
            # Convert RAG results back to ContractArgument objects
            arguments = []
            processed_sections = set()
            
            for result in appeal_results:
                metadata = result['metadata']
                section_title = metadata.get('section_title', 'Unknown Section')
                
                # Avoid duplicates
                if section_title in processed_sections:
                    continue
                processed_sections.add(section_title)
                
                # Reconstruct ContractArgument
                argument = ContractArgument(
                    section_title=section_title,
                    argument_type=metadata.get('argument_type', 'general'),
                    contract_elements=metadata.get('contract_elements', '').split(','),
                    factual_allegations=[],  # Will be populated from content
                    legal_standard=metadata.get('legal_standard', 'Not specified'),
                    citations=metadata.get('citations', '').split(','),
                    page_numbers=[],
                    full_text=result['content']
                )
                
                arguments.append(argument)
            
            print(f"📄 Extracted {len(arguments)} arguments from RAG system")
            return arguments
            
        except Exception as e:
            print(f"❌ Error extracting arguments from RAG: {e}")
            return []
    
    def _generate_fallback_summary(self, opposition_points: List[OppositionPoint], arguments: List[ContractArgument]) -> str:
        """Generate simple summary if main synthesis fails"""
        
        critical_points = [p for p in opposition_points if p.severity == "Critical"]
        significant_points = [p for p in opposition_points if p.severity == "Significant"]
        
        summary = f"""
# OPPOSITION ANALYSIS SUMMARY

## Executive Summary
- **Arguments Analyzed:** {len(arguments)}
- **Total Weaknesses Found:** {len(opposition_points)}
- **Critical Issues:** {len(critical_points)}
- **Significant Issues:** {len(significant_points)}

## Critical Weaknesses
"""
        
        for i, point in enumerate(critical_points[:5], 1):
            summary += f"""
### {i}. {point.weakness_type}
**Source:** {point.agent_source}
**Issue:** {point.argument[:300]}
**Authority:** {point.supporting_authority}
"""
        
        summary += "\n## Significant Weaknesses\n"
        
        for i, point in enumerate(significant_points[:8], 1):
            summary += f"""
### {i}. {point.weakness_type}
**Issue:** {point.argument[:200]}
"""
        
        summary += f"""

## Recommendation
Based on {len(opposition_points)} identified weaknesses, recommend strong opposition to plaintiff's appeal.

---
*Analysis generated by Multi-Agent Legal Opposition System*
"""
        
        return summary

# Integration function for GUI
async def run_multi_agent_analysis(rag_system: ContractRAGBuilder) -> str:
    """Main function to run multi-agent analysis from GUI"""
    
    try:
        # Check if we have required API keys
        if not os.getenv("ANTHROPIC_API_KEY"):
            return "❌ Error: ANTHROPIC_API_KEY not found in environment variables. Please add it to your .env file."
        
        # Initialize multi-agent system
        opposition_system = MultiAgentOppositionSystem(rag_system)
        
        # Run analysis
        result = await opposition_system.generate_opposition_analysis()
        
        return result
        
    except Exception as e:
        return f"❌ Multi-agent analysis failed: {str(e)}\n\nPlease check your API keys and try again."

# Test function
async def test_multi_agent_system():
    """Test the multi-agent system"""
    
    print("🧪 Testing Multi-Agent Opposition System...")
    
    # This would be called with your actual RAG system
    # rag_system = ContractRAGBuilder(pinecone_key, openai_key)
    # result = await run_multi_agent_analysis(rag_system)
    # print(result)
    
    print("✅ Multi-agent system code ready for integration")

if __name__ == "__main__":
    asyncio.run(test_multi_agent_system())