# Contract Analysis GUI with Multi-Agent Integration

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
import threading
import logging
import asyncio
from datetime import datetime
from typing import List, Dict

# Load environment variables
try:
   from dotenv import load_dotenv
   load_dotenv()
except ImportError:
   print("Install python-dotenv for .env file support: pip install python-dotenv")

# Import our contract analysis system
from contract_rag_foundation import ContractRAGBuilder, ContractDocumentProcessor
from multi_agent_analysis_system import run_multi_agent_analysis

class ContractAnalysisGUI:
   """GUI for contract law opposition analysis"""
   
   def __init__(self):
       self.root = tk.Tk()
       self.root.title("Contract Law Opposition Analysis System")
       self.root.geometry("900x700")
       
       # Initialize variables
       self.brief_file = None
       self.citation_files = []
       self.rag_builder = None
       
       # Setup logging to display in GUI
       self.setup_logging()
       
       # Create GUI components
       self.create_widgets()
       
       # Initialize RAG system
       self.initialize_rag_system()
   
   def setup_logging(self):
       """Setup logging to display in GUI"""
       self.log_handler = logging.Handler()
       self.log_handler.emit = self.log_to_gui
       
       logger = logging.getLogger()
       logger.addHandler(self.log_handler)
       logger.setLevel(logging.INFO)
   
   def log_to_gui(self, record):
       """Display log messages in GUI"""
       if hasattr(self, 'log_text'):
           self.log_text.insert(tk.END, f"{record.levelname}: {record.getMessage()}\n")
           self.log_text.see(tk.END)
           self.root.update_idletasks()
   
   def create_widgets(self):
       """Create main GUI components"""
       
       # Main frame
       main_frame = ttk.Frame(self.root, padding="10")
       main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
       
       # Title
       title_label = ttk.Label(main_frame, text="Contract Law Opposition Analysis", 
                              font=("Arial", 16, "bold"))
       title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
       
       # Document Selection Section
       doc_frame = ttk.LabelFrame(main_frame, text="Document Selection", padding="10")
       doc_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
       
       # Plaintiff's Brief Selection
       ttk.Label(doc_frame, text="Plaintiff's Appeal Brief (PDF):").grid(row=0, column=0, sticky=tk.W, pady=5)
       
       self.brief_path_var = tk.StringVar()
       self.brief_entry = ttk.Entry(doc_frame, textvariable=self.brief_path_var, width=60)
       self.brief_entry.grid(row=0, column=1, padx=(10, 5), pady=5)
       
       self.brief_button = ttk.Button(doc_frame, text="Browse", command=self.select_brief_file)
       self.brief_button.grid(row=0, column=2, padx=5, pady=5)
       
       # Citation Files Selection
       ttk.Label(doc_frame, text="Cited Cases Folder:").grid(row=1, column=0, sticky=tk.W, pady=5)
       
       self.citations_path_var = tk.StringVar()
       self.citations_entry = ttk.Entry(doc_frame, textvariable=self.citations_path_var, width=60)
       self.citations_entry.grid(row=1, column=1, padx=(10, 5), pady=5)
       
       self.citations_button = ttk.Button(doc_frame, text="Browse", command=self.select_citations_folder)
       self.citations_button.grid(row=1, column=2, padx=5, pady=5)
       
       # Selected Files Display
       files_frame = ttk.LabelFrame(main_frame, text="Selected Files", padding="10")
       files_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
       
       self.files_text = scrolledtext.ScrolledText(files_frame, height=8, width=80)
       self.files_text.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E))
       
       # Control Buttons
       button_frame = ttk.Frame(main_frame)
       button_frame.grid(row=3, column=0, columnspan=3, pady=10)
       
       self.process_button = ttk.Button(button_frame, text="Process Documents", 
                                       command=self.process_documents)
       self.process_button.grid(row=0, column=0, padx=5)
       
       self.analyze_button = ttk.Button(button_frame, text="Generate Opposition Analysis", 
                                       command=self.generate_analysis, state="disabled")
       self.analyze_button.grid(row=0, column=1, padx=5)
       
       self.clear_button = ttk.Button(button_frame, text="Clear All", command=self.clear_all)
       self.clear_button.grid(row=0, column=2, padx=5)
       
       # Progress Bar
       self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
       self.progress.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
       
       # Status and Log Section
       log_frame = ttk.LabelFrame(main_frame, text="Processing Log", padding="10")
       log_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
       
       self.log_text = scrolledtext.ScrolledText(log_frame, height=15, width=80)
       self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
       
       # Configure grid weights for resizing
       self.root.columnconfigure(0, weight=1)
       self.root.rowconfigure(0, weight=1)
       main_frame.columnconfigure(1, weight=1)
       main_frame.rowconfigure(5, weight=1)
       log_frame.columnconfigure(0, weight=1)
       log_frame.rowconfigure(0, weight=1)
       doc_frame.columnconfigure(1, weight=1)
       files_frame.columnconfigure(0, weight=1)
   
   def initialize_rag_system(self):
       """Initialize the RAG system with API keys"""
       try:
           # Get API keys from environment
           pinecone_key = os.getenv("PINECONE_API_KEY")
           openai_key = os.getenv("OPENAI_API_KEY")
           
           if not pinecone_key or not openai_key:
               messagebox.showerror("API Keys Missing", 
                                  "Please set PINECONE_API_KEY and OPENAI_API_KEY in your .env file")
               return
           
           self.rag_builder = ContractRAGBuilder(pinecone_key, openai_key)
           self.log_message("✅ RAG system initialized successfully")
           
       except Exception as e:
           messagebox.showerror("Initialization Error", f"Failed to initialize RAG system: {e}")
           self.log_message(f"❌ ERROR: Failed to initialize RAG system: {e}")
   
   def select_brief_file(self):
       """Select plaintiff's appeal brief PDF"""
       file_path = filedialog.askopenfilename(
           title="Select Plaintiff's Appeal Brief",
           filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
       )
       
       if file_path:
           self.brief_file = file_path
           self.brief_path_var.set(file_path)
           self.update_files_display()
           self.log_message(f"📄 Selected brief: {Path(file_path).name}")
   
   def select_citations_folder(self):
       """Select folder containing cited case PDFs"""
       folder_path = filedialog.askdirectory(
           title="Select Folder Containing Cited Cases (PDFs)"
       )
       
       if folder_path:
           # Find all PDF files in the folder
           pdf_files = list(Path(folder_path).glob("*.pdf"))
           
           if not pdf_files:
               messagebox.showwarning("No PDFs Found", "No PDF files found in selected folder")
               return
           
           self.citation_files = [str(pdf) for pdf in pdf_files]
           self.citations_path_var.set(folder_path)
           self.update_files_display()
           self.log_message(f"📁 Found {len(self.citation_files)} citation PDFs in {folder_path}")
   
   def update_files_display(self):
       """Update the files display area"""
       self.files_text.delete(1.0, tk.END)
       
       if self.brief_file:
           self.files_text.insert(tk.END, f"📄 PLAINTIFF'S BRIEF:\n")
           self.files_text.insert(tk.END, f"  {Path(self.brief_file).name}\n\n")
       
       if self.citation_files:
           self.files_text.insert(tk.END, f"📚 CITED CASES ({len(self.citation_files)} files):\n")
           for i, file_path in enumerate(self.citation_files, 1):
               self.files_text.insert(tk.END, f"  {i:2d}. {Path(file_path).name}\n")
   
   def process_documents(self):
       """Process all documents and build RAG system"""
       if not self.brief_file:
           messagebox.showerror("Missing Brief", "Please select the plaintiff's appeal brief")
           return
       
       if not self.citation_files:
           messagebox.showerror("Missing Citations", "Please select the folder containing cited cases")
           return
       
       if not self.rag_builder:
           messagebox.showerror("System Error", "RAG system not initialized")
           return
       
       # Run processing in separate thread to avoid blocking GUI
       self.process_button.config(state="disabled")
       self.progress.start()
       
       thread = threading.Thread(target=self._process_documents_thread)
       thread.daemon = True
       thread.start()
   
   def _process_documents_thread(self):
       """Process documents in background thread"""
       try:
           # Process plaintiff's brief
           self.log_message("🔄 Processing plaintiff's appeal brief...")
           
           # Extract text from brief PDF
           processor = ContractDocumentProcessor()
           brief_text = processor.extract_pdf_text(self.brief_file)
           
           if not brief_text:
               raise Exception("Failed to extract text from brief PDF")
           
           # Process and store brief
           arguments = self.rag_builder.process_and_store_appeal_brief(brief_text)
           self.log_message(f"✅ Brief processed: {len(arguments)} legal arguments identified")
           
           # Process cited cases
           self.log_message(f"🔄 Processing {len(self.citation_files)} cited cases...")
           cases = self.rag_builder.process_and_store_cited_cases(self.citation_files)
           self.log_message(f"✅ Processed {len(cases)} cited cases successfully")
           
           # Get RAG system stats
           stats = self.rag_builder.get_system_stats()
           self.log_message(f"🎯 RAG system ready: {stats['total_vectors']} vectors stored")
           
           # Enable analysis button
           self.root.after(0, self._processing_complete)
           
       except Exception as e:
           self.log_message(f"❌ ERROR: Processing failed: {e}")
           self.root.after(0, self._processing_failed)
   
   def _processing_complete(self):
       """Called when processing completes successfully"""
       self.progress.stop()
       self.process_button.config(state="normal")
       self.analyze_button.config(state="normal")
       messagebox.showinfo("Processing Complete", "Documents processed successfully! Ready for analysis.")
   
   def _processing_failed(self):
       """Called when processing fails"""
       self.progress.stop()
       self.process_button.config(state="normal")
       messagebox.showerror("Processing Failed", "Document processing failed. Check log for details.")
   
   def generate_analysis(self):
       """Generate opposition analysis using multi-agent system"""
       if not self.rag_builder:
           messagebox.showerror("System Error", "RAG system not ready")
           return
       
       # Check for Anthropic API key
       if not os.getenv("ANTHROPIC_API_KEY"):
           messagebox.showerror("Missing API Key", 
                              "ANTHROPIC_API_KEY not found. Please add it to your .env file.")
           return
       
       # Disable button and start progress
       self.analyze_button.config(state="disabled")
       self.progress.start()
       
       # Run multi-agent analysis in separate thread
       thread = threading.Thread(target=self._run_multi_agent_analysis)
       thread.daemon = True
       thread.start()
   
   def _run_multi_agent_analysis(self):
       """Run multi-agent analysis in background thread"""
       try:
           self.log_message("🚀 Starting multi-agent opposition analysis...")
           self.log_message("⚡ Deploying specialized legal analysis agents...")
           
           # Run the async multi-agent analysis
           loop = asyncio.new_event_loop()
           asyncio.set_event_loop(loop)
           
           result = loop.run_until_complete(run_multi_agent_analysis(self.rag_builder))
           loop.close()
           
           # Save result to file
           timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
           output_file = f"opposition_analysis_{timestamp}.txt"
           
           with open(output_file, 'w', encoding='utf-8') as f:
               f.write(result)
           
           self.log_message(f"✅ Opposition analysis complete!")
           self.log_message(f"📄 Report saved to: {output_file}")
           
           # Show completion in GUI thread
           self.root.after(0, lambda: self._analysis_complete(output_file, result))
           
       except Exception as e:
           self.log_message(f"❌ ERROR: Multi-agent analysis failed: {e}")
           self.root.after(0, self._analysis_failed)
   
   def _analysis_complete(self, output_file: str, result: str):
       """Called when multi-agent analysis completes"""
       self.progress.stop()
       self.analyze_button.config(state="normal")
       
       # Show results in a new window
       self._show_results_window(output_file, result)
       
       messagebox.showinfo("Analysis Complete", 
                          f"Opposition analysis complete!\n\nReport saved to: {output_file}")
   
   def _analysis_failed(self):
       """Called when analysis fails"""
       self.progress.stop()
       self.analyze_button.config(state="normal")
       messagebox.showerror("Analysis Failed", 
                          "Multi-agent analysis failed. Check log for details.")
   
   def _show_results_window(self, filename: str, content: str):
       """Show analysis results in new window"""
       
       # Create new window
       results_window = tk.Toplevel(self.root)
       results_window.title(f"Opposition Analysis Results - {filename}")
       results_window.geometry("1000x700")
       
       # Create text widget with scrollbar
       text_frame = ttk.Frame(results_window, padding="10")
       text_frame.pack(fill=tk.BOTH, expand=True)
       
       # Title
       title_label = ttk.Label(text_frame, text="Legal Opposition Analysis Report", 
                              font=("Arial", 14, "bold"))
       title_label.pack(pady=(0, 10))
       
       # Text area
       text_widget = scrolledtext.ScrolledText(text_frame, height=35, width=120, 
                                              font=("Consolas", 10))
       text_widget.pack(fill=tk.BOTH, expand=True)
       text_widget.insert(tk.END, content)
       text_widget.config(state=tk.DISABLED)  # Make read-only
       
       # Buttons
       button_frame = ttk.Frame(text_frame)
       button_frame.pack(pady=(10, 0))
       
       ttk.Button(button_frame, text="Save As...", 
                 command=lambda: self._save_as_dialog(content)).pack(side=tk.LEFT, padx=5)
       
       ttk.Button(button_frame, text="Copy to Clipboard", 
                 command=lambda: self._copy_to_clipboard(content)).pack(side=tk.LEFT, padx=5)
       
       ttk.Button(button_frame, text="Close", 
                 command=results_window.destroy).pack(side=tk.LEFT, padx=5)
   
   def _save_as_dialog(self, content: str):
       """Save analysis to user-selected file"""
       filename = filedialog.asksaveasfilename(
           title="Save Opposition Analysis",
           defaultextension=".txt",
           filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
       )
       
       if filename:
           try:
               with open(filename, 'w', encoding='utf-8') as f:
                   f.write(content)
               messagebox.showinfo("Saved", f"Analysis saved to {filename}")
           except Exception as e:
               messagebox.showerror("Save Error", f"Failed to save file: {e}")
   
   def _copy_to_clipboard(self, content: str):
       """Copy analysis to clipboard"""
       self.root.clipboard_clear()
       self.root.clipboard_append(content)
       messagebox.showinfo("Copied", "Analysis copied to clipboard")
   
   def clear_all(self):
       """Clear all selections and data"""
       self.brief_file = None
       self.citation_files = []
       self.brief_path_var.set("")
       self.citations_path_var.set("")
       self.files_text.delete(1.0, tk.END)
       self.log_text.delete(1.0, tk.END)
       self.analyze_button.config(state="disabled")
       self.log_message("🗑️ All data cleared")
   
   def log_message(self, message):
       """Add message to log"""
       if hasattr(self, 'log_text'):
           self.log_text.insert(tk.END, f"{message}\n")
           self.log_text.see(tk.END)
           self.root.update_idletasks()
   
   def run(self):
       """Start the GUI application"""
       self.root.mainloop()

def check_environment():
   """Check if required environment variables are set"""
   required_vars = ["PINECONE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
   missing_vars = []
   
   for var in required_vars:
       if not os.getenv(var):
           missing_vars.append(var)
   
   if missing_vars:
       print("❌ Missing required environment variables:")
       for var in missing_vars:
           print(f"   - {var}")
       print("\n📝 Create a .env file with:")
       for var in missing_vars:
           print(f"   {var}=your_key_here")
       return False
   
   print("✅ Environment variables configured")
   return True

def check_dependencies():
   """Check if required packages are installed"""
   required_packages = [
       "pinecone", "openai", "anthropic", "PyPDF2", "fitz", "dotenv"
   ]
   
   missing_packages = []
   
   for package in required_packages:
       try:
           if package == "fitz":
               import fitz
           elif package == "dotenv":
               from dotenv import load_dotenv
           else:
               __import__(package)
       except ImportError:
           missing_packages.append(package)
   
   if missing_packages:
       print("❌ Missing required packages:")
       package_map = {
           "fitz": "PyMuPDF",
           "dotenv": "python-dotenv"
       }
       
       for package in missing_packages:
           install_name = package_map.get(package, package)
           print(f"   pip install {install_name}")
       return False
   
   print("✅ All dependencies installed")
   return True

def setup_instructions():
   """Display setup instructions"""
   print("🎯 Contract Law Opposition Analysis System")
   print("=" * 50)
   print("\n📦 Quick Setup:")
   print("1. pip install pinecone openai anthropic PyPDF2 PyMuPDF python-dotenv")
   print("2. Create .env file with your API keys")
   print("3. python contract_gui_system.py")
   print("\n📁 Required documents:")
   print("   - Plaintiff's appeal brief (PDF)")
   print("   - Cited cases folder (15 PDF files)")
   print("\n🚀 Ready to analyze!")

if __name__ == "__main__":
   print("Starting Contract Law Opposition Analysis System...")
   
   # Show setup info
   setup_instructions()
   
   # Check dependencies
   if not check_dependencies():
       input("\nInstall missing packages and press Enter...")
   
   # Check environment
   if not check_environment():
       input("\nSet up .env file and press Enter...")
   
   try:
       # Start GUI
       print("\n🎯 Launching GUI...")
       app = ContractAnalysisGUI()
       app.run()
       
   except Exception as e:
       print(f"❌ Application error: {e}")
       input("Press Enter to exit...")