#!/usr/bin/env python3
"""
LaTeX Paper Compilation Script with Version Tracking
Compiles the IVF paper with automatic versioning and change tracking
"""

import os
import sys
import subprocess
import json
import shutil
import hashlib
from datetime import datetime
from pathlib import Path

class PaperCompiler:
    def __init__(self, tex_file="paper_latex_sn/main.tex", output_dir="compiled_papers"):
        self.tex_file = Path(tex_file)
        self.output_dir = Path(output_dir)
        self.version_file = Path("compilation_history.json")
        self.paper_name = "ART_Counseling_Framework"
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(exist_ok=True)
        
        # Load or initialize version history
        self.load_version_history()
    
    def load_version_history(self):
        """Load compilation history from JSON file"""
        if self.version_file.exists():
            with open(self.version_file, 'r') as f:
                self.history = json.load(f)
        else:
            self.history = {
                "current_version": "0.0.0",
                "compilations": [],
                "paper_name": self.paper_name
            }
    
    def save_version_history(self):
        """Save compilation history to JSON file"""
        with open(self.version_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def increment_version(self, version_type="patch"):
        """Increment version number (major.minor.patch)"""
        current = self.history["current_version"]
        major, minor, patch = map(int, current.split('.'))
        
        if version_type == "major":
            major += 1
            minor = 0
            patch = 0
        elif version_type == "minor":
            minor += 1
            patch = 0
        else:  # patch
            patch += 1
        
        new_version = f"{major}.{minor}.{patch}"
        self.history["current_version"] = new_version
        return new_version
    
    def get_file_hash(self, file_path):
        """Calculate MD5 hash of file content"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def collect_source_hashes(self):
        """Collect hashes of all source files to detect changes"""
        source_files = [
            self.tex_file,
            self.tex_file.parent / "references.bib",
            self.tex_file.parent / "sections" / "introduction.tex",
            self.tex_file.parent / "sections" / "methods.tex",
            self.tex_file.parent / "sections" / "results.tex",
            self.tex_file.parent / "sections" / "discussion.tex",
            self.tex_file.parent / "sections" / "conclusion.tex",
        ]
        
        hashes = {}
        for file_path in source_files:
            if file_path.exists():
                hashes[str(file_path)] = self.get_file_hash(file_path)
        
        return hashes
    
    def has_changes(self, new_hashes):
        """Check if there are changes since last compilation"""
        if not self.history["compilations"]:
            return True
        
        last_compilation = self.history["compilations"][-1]
        last_hashes = last_compilation.get("source_hashes", {})
        
        return new_hashes != last_hashes
    
    def run_latex_compilation(self, version):
        """Run pdflatex compilation with proper error handling"""
        print(f"🔄 Compiling LaTeX document: {self.tex_file}")
        
        # Change to document directory for compilation
        original_dir = os.getcwd()
        tex_dir = self.tex_file.parent
        os.chdir(tex_dir)
        
        # Generate timestamped filename for output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"{self.paper_name}_v{version}_{timestamp}"
        
        try:
            # Run pdflatex multiple times for proper reference resolution
            commands = [
                ["pdflatex", f"-jobname={output_name}", "-interaction=nonstopmode", "main.tex"],
                ["bibtex", output_name],
                ["pdflatex", f"-jobname={output_name}", "-interaction=nonstopmode", "main.tex"],
                ["pdflatex", f"-jobname={output_name}", "-interaction=nonstopmode", "main.tex"]
            ]
            
            for i, cmd in enumerate(commands):
                print(f"  Step {i+1}/4: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    print(f"❌ Error in step {i+1}:")
                    print(result.stdout)
                    print(result.stderr)
                    if i > 1:  # Allow bibtex to fail gracefully
                        continue
                    else:
                        raise subprocess.CalledProcessError(result.returncode, cmd)
            
            return output_name
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Compilation failed: {e}")
            return None
        finally:
            os.chdir(original_dir)
    
    def create_raw_tex_file(self, version):
        """Create a single concatenated tex file with all sections and bibliography"""
        print(f"📝 Creating raw tex file...")
        
        tex_dir = self.tex_file.parent
        main_tex_content = ""
        
        # Read main.tex file
        with open(self.tex_file, 'r', encoding='utf-8') as f:
            main_content = f.read()
        
        # Process line by line to handle \input and \bibliography commands
        lines = main_content.split('\n')
        processed_lines = []
        
        for line in lines:
            if line.strip().startswith('\\input{'):
                # Extract filename from \input{filename}
                input_file = line.strip()[7:-1]  # Remove \input{ and }
                input_path = tex_dir / f"{input_file}.tex"
                
                if input_path.exists():
                    print(f"  📄 Including: {input_file}.tex")
                    with open(input_path, 'r', encoding='utf-8') as f:
                        section_content = f.read()
                    processed_lines.append(f"% === INCLUDED FROM {input_file}.tex ===")
                    processed_lines.append(section_content)
                    processed_lines.append(f"% === END OF {input_file}.tex ===")
                    processed_lines.append("")
                else:
                    processed_lines.append(line)
                    
            elif line.strip().startswith('\\bibliography{'):
                # For the raw tex file, we'll include the original .bib content as comments
                # and keep the \bibliography command for proper processing
                bib_file = line.strip()[13:-1]  # Remove \bibliography{ and }
                bib_path = tex_dir / f"{bib_file}.bib"
                
                if bib_path.exists():
                    print(f"  📚 Including: {bib_file}.bib as reference")
                    with open(bib_path, 'r', encoding='utf-8') as f:
                        bib_content = f.read()
                    
                    processed_lines.append(f"% === BIBLIOGRAPHY CONTENT FROM {bib_file}.bib ===")
                    processed_lines.append("% The following is the bibliography content for reference:")
                    # Add each line of bib content as a comment
                    for bib_line in bib_content.split('\n'):
                        processed_lines.append(f"% {bib_line}")
                    processed_lines.append("% === END OF BIBLIOGRAPHY CONTENT ===")
                    processed_lines.append("")
                    processed_lines.append("% Keep original bibliography command:")
                    processed_lines.append(line)
                else:
                    processed_lines.append(line)
            else:
                processed_lines.append(line)
        
        # Join all processed lines
        final_content = '\n'.join(processed_lines)
        
        # Create raw tex filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_tex_filename = f"{self.paper_name}_v{version}_{timestamp}_raw.tex"
        raw_tex_path = self.output_dir / raw_tex_filename
        
        # Write the concatenated file
        with open(raw_tex_path, 'w', encoding='utf-8') as f:
            f.write(final_content)
        
        print(f"📝 Raw tex file created: {raw_tex_filename}")
        return raw_tex_path

    def clean_auxiliary_files(self, output_name=None):
        """Clean up auxiliary LaTeX files"""
        tex_dir = self.tex_file.parent
        aux_extensions = ['.aux', '.log', '.bbl', '.blg', '.out', '.toc', '.synctex.gz']
        
        # Clean files for the specific output name if provided
        if output_name:
            for ext in aux_extensions:
                aux_file = tex_dir / f"{output_name}{ext}"
                if aux_file.exists():
                    aux_file.unlink()
                    print(f"🧹 Cleaned: {aux_file.name}")
        
        # Also clean any remaining main.* files
        for ext in aux_extensions:
            aux_file = tex_dir / f"main{ext}"
            if aux_file.exists():
                aux_file.unlink()
                print(f"🧹 Cleaned: {aux_file.name}")
    
    def copy_pdf_with_version(self, output_name):
        """Copy compiled PDF to output directory with version name"""
        source_pdf = self.tex_file.parent / f"{output_name}.pdf"
        
        if not source_pdf.exists():
            raise FileNotFoundError(f"Compiled PDF not found: {source_pdf}")
        
        # Copy PDF to versioned location
        filename = f"{output_name}.pdf"
        destination = self.output_dir / filename
        shutil.copy2(source_pdf, destination)
        
        # Also create a "latest" symlink or copy
        latest_path = self.output_dir / f"{self.paper_name}_latest.pdf"
        if latest_path.exists():
            latest_path.unlink()
        shutil.copy2(source_pdf, latest_path)
        
        return destination, latest_path
    
    def compile(self, version_type="patch", force=False):
        """Main compilation function"""
        print(f"🚀 Starting compilation of {self.paper_name}")
        print(f"📄 Source: {self.tex_file}")
        
        # Check for changes
        source_hashes = self.collect_source_hashes()
        
        if not force and not self.has_changes(source_hashes):
            print("ℹ️  No changes detected since last compilation")
            print("   Use --force to compile anyway")
            return False
        
        # Increment version
        new_version = self.increment_version(version_type)
        print(f"📈 Version: {new_version}")
        
        # Compile LaTeX
        compilation_start = datetime.now()
        output_name = self.run_latex_compilation(new_version)
        
        if not output_name:
            print("❌ Compilation failed!")
            return False
        
        compilation_end = datetime.now()
        compilation_time = (compilation_end - compilation_start).total_seconds()
        
        try:
            # Copy PDF with version
            versioned_pdf, latest_pdf = self.copy_pdf_with_version(output_name)
            
            # Create raw tex file
            raw_tex_path = self.create_raw_tex_file(new_version)
            
            # Record compilation in history
            compilation_record = {
                "version": new_version,
                "timestamp": compilation_end.isoformat(),
                "compilation_time_seconds": compilation_time,
                "versioned_file": str(versioned_pdf),
                "latest_file": str(latest_pdf),
                "raw_tex_file": str(raw_tex_path),
                "source_hashes": source_hashes,
                "tex_file": str(self.tex_file),
                "success": True
            }
            
            self.history["compilations"].append(compilation_record)
            self.save_version_history()
            
            # Clean up auxiliary files
            self.clean_auxiliary_files(output_name)
            
            print(f"✅ Compilation successful!")
            print(f"📁 Versioned PDF: {versioned_pdf}")
            print(f"📁 Latest PDF: {latest_pdf}")
            print(f"📝 Raw TeX file: {raw_tex_path}")
            print(f"⏱️  Compilation time: {compilation_time:.1f}s")
            
            return True
            
        except Exception as e:
            print(f"❌ Error during post-compilation: {e}")
            return False

def main():
    """Main function with command line argument handling"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compile LaTeX paper with version tracking")
    parser.add_argument("--version-type", choices=["major", "minor", "patch"], 
                       default="patch", help="Type of version increment")
    parser.add_argument("--force", action="store_true", 
                       help="Force compilation even if no changes detected")
    parser.add_argument("--tex-file", default="paper_latex_sn/main.tex",
                       help="Path to main LaTeX file")
    parser.add_argument("--output-dir", default="compiled_papers",
                       help="Output directory for compiled PDFs")
    
    args = parser.parse_args()
    
    # Create compiler instance
    compiler = PaperCompiler(args.tex_file, args.output_dir)
    
    # Run compilation
    success = compiler.compile(args.version_type, args.force)
    
    if success:
        print(f"\n📊 Compilation History:")
        print(f"   Total compilations: {len(compiler.history['compilations'])}")
        print(f"   Current version: {compiler.history['current_version']}")
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main() 