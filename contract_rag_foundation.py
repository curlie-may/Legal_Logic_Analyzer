# Contract Law RAG Foundation - Implementation Priority 1

import os
import re
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import logging

# Core dependencies
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
import PyPDF2
import fitz  # PyMuPDF for better PDF handling

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ContractArgument:
   """Structured representation of contract law argument"""
   section_title: str
   argument_type: str  # "formation", "breach", "damages", "defenses"
   contract_elements: List[str]  # Which elements this addresses
   factual_allegations: List[str]
   legal_standard: str
   citations: List[str]
   page_numbers: List[int]
   full_text: str

@dataclass
class CaseData:
   """Structured representation of cited case"""
   citation: str
   case_name: str
   court: str
   year: str
   facts_section: str
   holding_section: str
   reasoning_section: str
   contract_principles: List[str]
   full_text: str

class ContractDocumentProcessor:
   """Processes contract law documents with legal structure awareness"""
   
   def __init__(self):
       self.contract_elements = {
           'formation': ['offer', 'acceptance', 'consideration', 'mutual assent', 'intent'],
           'performance': ['substantial performance', 'material breach', 'minor breach'],
           'interpretation': ['parol evidence', 'ambiguity', 'plain meaning', 'extrinsic evidence'],
           'defenses': ['unconscionability', 'statute of limitations', 'impossibility', 'frustration'],
           'damages': ['expectation', 'consequential', 'incidental', 'liquidated', 'mitigation']
       }
       
       self.citation_patterns = [
           r'\d+\s+[A-Z][a-z\.]+\s+\d+.*?\(\d{4}\)',  # Standard citation
           r'\d+\s+F\.\d+d\s+\d+.*?\([A-Za-z\.\s]+\d{4}\)',  # Federal citation
           r'\d+\s+U\.S\.\s+\d+.*?\(\d{4}\)',  # Supreme Court
       ]
       
       self.legal_headers = [
           r'STATEMENT OF FACTS',
           r'PROCEDURAL HISTORY', 
           r'STANDARD OF REVIEW',
           r'ARGUMENT',
           r'CONCLUSION',
           r'I\.\s+',  # Roman numerals
           r'A\.\s+',  # Letter subsections
           r'1\.\s+'   # Numbered subsections
       ]
   
   def extract_pdf_text(self, pdf_path: str) -> str:
       """Extract text from PDF with better formatting preservation"""
       
       try:
           # Try PyMuPDF first (better formatting)
           doc = fitz.open(pdf_path)
           text = ""
           
           for page_num in range(len(doc)):
               page = doc.load_page(page_num)
               text += f"\n--- PAGE {page_num + 1} ---\n"
               text += page.get_text()
           
           doc.close()
           return text
           
       except Exception as e:
           logger.warning(f"PyMuPDF failed for {pdf_path}, trying PyPDF2: {e}")
           
           # Fallback to PyPDF2
           try:
               with open(pdf_path, 'rb') as file:
                   pdf_reader = PyPDF2.PdfReader(file)
                   text = ""
                   
                   for page_num, page in enumerate(pdf_reader.pages):
                       text += f"\n--- PAGE {page_num + 1} ---\n"
                       text += page.extract_text()
                   
                   return text
                   
           except Exception as e2:
               logger.error(f"Failed to extract text from {pdf_path}: {e2}")
               return ""
   
   def parse_appeal_brief(self, brief_text: str) -> List[ContractArgument]:
       """Parse 25-page appeal brief into structured arguments"""
       
       logger.info("Parsing appeal brief structure...")
       
       # 1. Split into major sections
       sections = self._split_by_legal_headers(brief_text)
       
       # 2. Extract contract-specific arguments
       contract_arguments = []
       
       for section_title, section_text in sections.items():
           
           # Skip non-argument sections
           if any(skip in section_title.upper() for skip in ['FACTS', 'PROCEDURAL', 'CONCLUSION']):
               continue
           
           # Determine argument type
           arg_type = self._classify_argument_type(section_text)
           
           # Extract contract elements addressed
           elements = self._extract_contract_elements(section_text, arg_type)
           
           # Extract factual allegations
           facts = self._extract_factual_allegations(section_text)
           
           # Extract legal standard
           legal_standard = self._extract_legal_standard(section_text)
           
           # Extract citations
           citations = self._extract_citations(section_text)
           
           # Extract page numbers
           page_numbers = self._extract_page_references(section_text)
           
           contract_arg = ContractArgument(
               section_title=section_title,
               argument_type=arg_type,
               contract_elements=elements,
               factual_allegations=facts,
               legal_standard=legal_standard,
               citations=citations,
               page_numbers=page_numbers,
               full_text=section_text
           )
           
           contract_arguments.append(contract_arg)
           
           logger.info(f"Parsed argument: {section_title} ({arg_type})")
       
       return contract_arguments
   
   def _split_by_legal_headers(self, text: str) -> Dict[str, str]:
       """Split document by legal section headers"""
       
       sections = {}
       
       # Find all header positions
       header_positions = []
       for pattern in self.legal_headers:
           matches = list(re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE))
           for match in matches:
               header_positions.append((match.start(), match.group(), match.end()))
       
       # Sort by position
       header_positions.sort(key=lambda x: x[0])
       
       # Extract sections
       for i, (start_pos, header, header_end) in enumerate(header_positions):
           end_pos = header_positions[i + 1][0] if i + 1 < len(header_positions) else len(text)
           section_text = text[header_end:end_pos].strip()
           
           if section_text:  # Only add non-empty sections
               sections[header.strip()] = section_text
       
       return sections
   
   def _classify_argument_type(self, section_text: str) -> str:
       """Determine contract argument type from content"""
       
       text_lower = section_text.lower()
       
       # Check for formation arguments
       formation_indicators = ['offer', 'acceptance', 'consideration', 'mutual assent', 'formation']
       if any(indicator in text_lower for indicator in formation_indicators):
           return 'formation'
       
       # Check for breach arguments
       breach_indicators = ['breach', 'performance', 'material breach', 'substantial performance']
       if any(indicator in text_lower for indicator in breach_indicators):
           return 'breach'
       
       # Check for damages arguments
       damages_indicators = ['damages', 'consequential', 'incidental', 'expectation', 'lost profits']
       if any(indicator in text_lower for indicator in damages_indicators):
           return 'damages'
       
       # Check for defenses
       defense_indicators = ['unconscionable', 'statute of limitations', 'parol evidence', 'impossibility']
       if any(indicator in text_lower for indicator in defense_indicators):
           return 'defenses'
       
       # Check for interpretation issues
       interpretation_indicators = ['interpretation', 'ambiguous', 'plain meaning', 'extrinsic evidence']
       if any(indicator in text_lower for indicator in interpretation_indicators):
           return 'interpretation'
       
       return 'general'
   
   def _extract_contract_elements(self, text: str, arg_type: str) -> List[str]:
       """Extract specific contract elements addressed in this argument"""
       
       elements_found = []
       text_lower = text.lower()
       
       if arg_type in self.contract_elements:
           for element in self.contract_elements[arg_type]:
               if element in text_lower:
                   elements_found.append(element)
       
       return elements_found
   
   def _extract_factual_allegations(self, text: str) -> List[str]:
       """Extract factual allegations (key for de novo analysis)"""
       
       factual_allegations = []
       
       # Look for fact-indicating patterns
       fact_patterns = [
           r'[Pp]laintiff alleges that ([^.]+\.)',
           r'[Tt]he facts show that ([^.]+\.)',
           r'[Tt]he record establishes ([^.]+\.)',
           r'[Ii]t is undisputed that ([^.]+\.)',
           r'[Tt]he complaint alleges ([^.]+\.)'
       ]
       
       for pattern in fact_patterns:
           matches = re.findall(pattern, text)
           factual_allegations.extend(matches)
       
       # Also look for sentences with factual indicators
       sentences = re.split(r'[.!?]+', text)
       for sentence in sentences:
           if any(indicator in sentence.lower() for indicator in ['alleges', 'claims', 'asserts', 'establishes']):
               if len(sentence.strip()) > 20:  # Avoid very short sentences
                   factual_allegations.append(sentence.strip())
       
       return factual_allegations[:10]  # Limit to most relevant
   
   def _extract_legal_standard(self, text: str) -> str:
       """Extract the legal standard being applied"""
       
       standard_patterns = [
           r'[Uu]nder\s+([^,]{10,50})',
           r'[Tt]he standard is ([^.]{10,100})',
           r'[Tt]o establish ([^,]{10,80})',
           r'Rule 12\(b\)\(6\) requires ([^.]{10,100})',
           r'De novo review ([^.]{10,100})'
       ]
       
       for pattern in standard_patterns:
           match = re.search(pattern, text)
           if match:
               return match.group(1).strip()
       
       return 'Standard not clearly identified'
   
   def _extract_citations(self, text: str) -> List[str]:
       """Extract all legal citations from text"""
       
       citations = []
       
       for pattern in self.citation_patterns:
           matches = re.findall(pattern, text)
           citations.extend(matches)
       
       # Clean and deduplicate
       cleaned_citations = []
       for citation in citations:
           cleaned = citation.strip()
           if cleaned and cleaned not in cleaned_citations:
               cleaned_citations.append(cleaned)
       
       return cleaned_citations
   
   def _extract_page_references(self, text: str) -> List[int]:
       """Extract page number references"""
       
       page_refs = []
       
       # Look for page references in text
       page_patterns = [
           r'--- PAGE (\d+) ---',
           r'page (\d+)',
           r'p\. (\d+)',
           r'\[(\d+)\]'  # Bracket page references
       ]
       
       for pattern in page_patterns:
           matches = re.findall(pattern, text, re.IGNORECASE)
           for match in matches:
               try:
                   page_refs.append(int(match))
               except ValueError:
                   continue
       
       return sorted(list(set(page_refs)))
   
   def parse_cited_case(self, case_text: str, pdf_filename: str) -> CaseData:
       """Parse cited case into structured format"""
       
       logger.info(f"Parsing cited case: {pdf_filename}")
       
       # Extract basic case information
       citation = self._extract_case_citation(case_text, pdf_filename)
       case_name = self._extract_case_name(case_text)
       court = self._extract_court(case_text)
       year = self._extract_year(case_text)
       
       # Extract major sections
       facts_section = self._extract_facts_section(case_text)
       holding_section = self._extract_holding_section(case_text)
       reasoning_section = self._extract_reasoning_section(case_text)
       
       # Extract contract law principles
       contract_principles = self._extract_contract_principles(case_text)
       
       return CaseData(
           citation=citation,
           case_name=case_name,
           court=court,
           year=year,
           facts_section=facts_section,
           holding_section=holding_section,
           reasoning_section=reasoning_section,
           contract_principles=contract_principles,
           full_text=case_text
       )
   
   def _extract_case_citation(self, text: str, filename: str) -> str:
       """Extract primary case citation"""
       
       # Try to find citation in text
       for pattern in self.citation_patterns:
           match = re.search(pattern, text)
           if match:
               return match.group()
       
       # Fallback to filename
       return Path(filename).stem.replace('_', ' ')
   
   def _extract_case_name(self, text: str) -> str:
       """Extract case name (e.g., 'Smith v. Jones')"""
       
       # Look for case name patterns
       name_patterns = [
           r'([A-Z][a-z]+\s+v\.\s+[A-Z][a-z]+)',
           r'([A-Z][A-Z\s]+v\.\s+[A-Z][A-Z\s]+)',
       ]
       
       for pattern in name_patterns:
           match = re.search(pattern, text)
           if match:
               return match.group(1)
       
       return "Case name not found"
   
   def _extract_court(self, text: str) -> str:
       """Extract court information"""
       court_patterns = [
           r'United States Supreme Court',
           r'U\.S\. Supreme Court',
           r'Court of Appeals',
           r'District Court',
           r'Supreme Court'
       ]
       
       for pattern in court_patterns:
           if re.search(pattern, text, re.IGNORECASE):
               return pattern
       
       return "Court not identified"
   
   def _extract_year(self, text: str) -> str:
       """Extract year from case"""
       year_match = re.search(r'\b(19|20)\d{2}\b', text)
       return year_match.group() if year_match else "Year not found"
   
   def _extract_facts_section(self, text: str) -> str:
       """Extract facts section from case"""
       
       # Look for facts section markers
       facts_markers = ['FACTS', 'BACKGROUND', 'FACTUAL BACKGROUND']
       
       for marker in facts_markers:
           pattern = rf'{marker}(.*?)(?=DISCUSSION|ANALYSIS|HOLDING|CONCLUSION)'
           match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
           if match:
               return match.group(1).strip()
       
       # If no clear section, extract first substantial paragraph
       paragraphs = text.split('\n\n')
       for para in paragraphs:
           if len(para) > 200:  # Substantial paragraph
               return para.strip()
       
       return "Facts section not clearly identified"
   
   def _extract_holding_section(self, text: str) -> str:
       """Extract holding section"""
       holding_patterns = [
           r'HOLDING(.*?)(?=DISCUSSION|ANALYSIS|CONCLUSION)',
           r'We hold that(.*?)(?=\.|$)',
           r'The court holds(.*?)(?=\.|$)'
       ]
       
       for pattern in holding_patterns:
           match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
           if match:
               return match.group(1).strip()
       
       return "Holding not clearly identified"
   
   def _extract_reasoning_section(self, text: str) -> str:
       """Extract reasoning/analysis section"""
       reasoning_patterns = [
           r'DISCUSSION(.*?)(?=CONCLUSION|$)',
           r'ANALYSIS(.*?)(?=CONCLUSION|$)',
           r'REASONING(.*?)(?=CONCLUSION|$)'
       ]
       
       for pattern in reasoning_patterns:
           match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
           if match:
               return match.group(1).strip()
       
       return "Reasoning section not clearly identified"
   
   def _extract_contract_principles(self, text: str) -> List[str]:
       """Extract contract law principles from case"""
       
       principles = []
       text_lower = text.lower()
       
       # Common contract law principles
       principle_indicators = {
           'offer_acceptance': ['offer and acceptance', 'mutual assent', 'meeting of minds'],
           'consideration': ['consideration', 'bargained-for exchange', 'benefit and detriment'],
           'breach': ['material breach', 'substantial performance', 'fundamental breach'],
           'damages': ['expectation damages', 'consequential damages', 'foreseeability'],
           'interpretation': ['parol evidence rule', 'plain meaning', 'ambiguity'],
           'defenses': ['unconscionability', 'statute of limitations', 'impossibility']
       }
       
       for principle_type, indicators in principle_indicators.items():
           for indicator in indicators:
               if indicator in text_lower:
                   principles.append(f"{principle_type}: {indicator}")
       
       return principles

class ContractRAGBuilder:
   """Builds RAG system for contract law documents"""
   
   def __init__(self, pinecone_key: str, openai_key: str):
       # Initialize Pinecone with new API
       self.pc = Pinecone(api_key=pinecone_key)
       
       self.index_name = "contract-law-rag"
       
       # Create index if needed
       if self.index_name not in [index.name for index in self.pc.list_indexes()]:
           logger.info(f"Creating Pinecone index: {self.index_name}")
           self.pc.create_index(
               name=self.index_name,
               dimension=3072,  # text-embedding-3-large
               metric="cosine",
               spec=ServerlessSpec(
                   cloud="aws",
                   region="us-east-1"
               )
           )
       
       self.index = self.pc.Index(self.index_name)
       self.openai_client = OpenAI(api_key=openai_key)
       self.processor = ContractDocumentProcessor()
       
       logger.info("Contract RAG system initialized")
   
   def embed_text(self, text: str) -> List[float]:
       """Generate embeddings using OpenAI"""
       
       # Truncate if too long
       if len(text) > 8000:
           text = text[:8000]
       
       try:
           response = self.openai_client.embeddings.create(
               model="text-embedding-3-large",
               input=text
           )
           return response.data[0].embedding
       
       except Exception as e:
           logger.error(f"Embedding failed: {e}")
           return []
   
   def process_and_store_appeal_brief(self, brief_text: str) -> List[ContractArgument]:
       """Process appeal brief and store in RAG"""
       
       logger.info("Processing appeal brief...")
       
       # Parse brief structure
       arguments = self.processor.parse_appeal_brief(brief_text)
       
       # Create chunks for each argument
       vectors_to_store = []
       
       for i, argument in enumerate(arguments):
           
           # Create embedding for full argument
           embedding = self.embed_text(argument.full_text)
           
           if not embedding:
               continue
           
           # Prepare metadata
           metadata = {
               'document_type': 'appeal_brief',
               'argument_type': argument.argument_type,
               'contract_elements': ','.join(argument.contract_elements),
               'legal_standard': argument.legal_standard,
               'section_title': argument.section_title,
               'citations': ','.join(argument.citations[:5]),  # Limit size
               'factual_allegations_count': len(argument.factual_allegations),
               'content_preview': argument.full_text[:500]
           }
           
           vector = {
               'id': f"appeal_arg_{i}",
               'values': embedding,
               'metadata': metadata
           }
           
           vectors_to_store.append(vector)
       
       # Store in Pinecone
       if vectors_to_store:
           self.index.upsert(vectors=vectors_to_store)
           logger.info(f"Stored {len(vectors_to_store)} appeal arguments in RAG")
       
       return arguments
   
   def process_and_store_cited_cases(self, case_pdf_files: List[str]) -> List[CaseData]:
       """Process cited cases and store in RAG"""
       
       logger.info(f"Processing {len(case_pdf_files)} cited cases...")
       
       processed_cases = []
       vectors_to_store = []
       
       for pdf_file in case_pdf_files:
           try:
               # Extract case text
               case_text = self.processor.extract_pdf_text(pdf_file)
               
               if not case_text:
                   logger.warning(f"No text extracted from {pdf_file}")
                   continue
               
               # Parse case structure
               case_data = self.processor.parse_cited_case(case_text, pdf_file)
               processed_cases.append(case_data)
               
               # Create multiple embeddings for different sections
               sections = {
                   'facts': case_data.facts_section,
                   'holding': case_data.holding_section,
                   'reasoning': case_data.reasoning_section,
                   'full_case': case_data.full_text[:8000]  # Truncate if needed
               }
               
               for section_name, section_text in sections.items():
                   if len(section_text) < 50:  # Skip very short sections
                       continue
                   
                   embedding = self.embed_text(section_text)
                   
                   if not embedding:
                       continue
                   
                   # Prepare metadata
                   metadata = {
                       'document_type': 'cited_case',
                       'case_name': case_data.case_name,
                       'citation': case_data.citation,
                       'court': case_data.court,
                       'year': case_data.year,
                       'section_type': section_name,
                       'contract_principles': ','.join(case_data.contract_principles[:3]),
                       'content_preview': section_text[:500],
                       'pdf_source': Path(pdf_file).name
                   }
                   
                   vector = {
                       'id': f"case_{Path(pdf_file).stem}_{section_name}",
                       'values': embedding,
                       'metadata': metadata
                   }
                   
                   vectors_to_store.append(vector)
               
               logger.info(f"Processed case: {case_data.case_name}")
               
           except Exception as e:
               logger.error(f"Failed to process {pdf_file}: {e}")
               continue
       
       # Store all vectors in Pinecone
       if vectors_to_store:
           # Batch upsert
           batch_size = 100
           for i in range(0, len(vectors_to_store), batch_size):
               batch = vectors_to_store[i:i + batch_size]
               self.index.upsert(vectors=batch)
               logger.info(f"Stored batch {i//batch_size + 1}")
           
           logger.info(f"Stored {len(vectors_to_store)} case sections in RAG")
       
       return processed_cases
   
   def search_contract_precedent(self, query: str, argument_type: str = None, 
                                top_k: int = 10) -> List[Dict]:
       """Search for relevant contract law precedent"""
       
       # Generate query embedding
       query_embedding = self.embed_text(query)
       
       if not query_embedding:
           return []
       
       # Build search filters
       filters = {}
       if argument_type:
           filters['argument_type'] = argument_type
       
       # Search vector database
       results = self.index.query(
           vector=query_embedding,
           top_k=top_k,
           include_metadata=True,
           filter=filters
       )
       
       # Format results
       formatted_results = []
       for match in results['matches']:
           formatted_results.append({
               'score': match['score'],
               'content': match['metadata'].get('content_preview', ''),
               'metadata': match['metadata'],
               'document_type': match['metadata'].get('document_type'),
               'relevance': 'High' if match['score'] > 0.8 else 'Medium' if match['score'] > 0.7 else 'Low'
           })
       
       return formatted_results
   
   def get_system_stats(self) -> Dict:
       """Get RAG system statistics"""
       
       stats = self.index.describe_index_stats()
       
       return {
           'total_vectors': stats.total_vector_count,
           'index_fullness': stats.index_fullness,
           'namespaces': stats.namespaces
       }

# Test and validation functions
def test_rag_system():
   """Test the RAG system with sample data"""
   
   # Initialize
   rag_builder = ContractRAGBuilder(
       pinecone_key=os.getenv("PINECONE_API_KEY"),
       openai_key=os.getenv("OPENAI_API_KEY")
   )
   
   # Test search
   test_queries = [
       "contract formation offer acceptance consideration",
       "material breach substantial performance",
       "expectation damages foreseeability",
       "parol evidence rule contract interpretation"
   ]
   
   print("Testing RAG search capabilities...")
   
   for query in test_queries:
       results = rag_builder.search_contract_precedent(query, top_k=3)
       print(f"\nQuery: {query}")
       print(f"Results found: {len(results)}")
       
       for i, result in enumerate(results[:2]):
           print(f"  {i+1}. Score: {result['score']:.3f}")
           print(f"     Type: {result['document_type']}")
           print(f"     Preview: {result['content'][:100]}...")
   
   # Get stats
   stats = rag_builder.get_system_stats()
   print(f"\nRAG System Stats: {stats}")

if __name__ == "__main__":
   # Example usage
   print("Contract Law RAG Foundation - Ready for Implementation")
   print("Next steps:")
   print("1. Set environment variables: PINECONE_API_KEY, OPENAI_API_KEY")
   print("2. Run: python contract_rag_foundation.py")
   print("3. Test with your appeal brief and cited cases")
   
   # Uncomment to run test
   # test_rag_system()