"""
Step 1: Extract Text from Government Scheme PDFs
================================================
Updated for: Farming Scemed.pdf and farmerbook.pdf
"""

import os
import PyPDF2
from pathlib import Path

# =============================================
# CONFIGURATION
# =============================================
PDF_FOLDER = "../data/schemes/"  # Folder containing PDFs
OUTPUT_FOLDER = "../data/processed/"  # Where to save extracted text
OUTPUT_FILE = "../data/processed/schemes_text.txt"  # Combined text file

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# =============================================
# FUNCTION: Extract Text from Single PDF
# =============================================
def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file
    Returns: extracted text as string
    """
    text = ""
    
    try:
        print(f"   Opening PDF...")
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            
            print(f"   Total pages: {num_pages}")
            
            for page_num in range(num_pages):
                try:
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    
                    if page_text and page_text.strip():
                        text += page_text
                        text += "\n\n--- Page Break ---\n\n"
                        print(f"   ✓ Extracted page {page_num + 1}/{num_pages}")
                    else:
                        print(f"   ⚠️  Page {page_num + 1} appears empty")
                except Exception as e:
                    print(f"   ⚠️  Error on page {page_num + 1}: {e}")
                    continue
        
        if not text.strip():
            print(f"   ⚠️  No text extracted (might be scanned images)")
            return ""
        
        return text
    
    except Exception as e:
        print(f"   ❌ Error extracting {pdf_path.name}: {e}")
        return ""

# =============================================
# FUNCTION: Clean Extracted Text
# =============================================
def clean_text(text):
    """
    Clean extracted text - remove extra whitespace, special characters
    """
    # Remove multiple newlines
    lines = text.split('\n')
    cleaned_lines = [line.strip() for line in lines if line.strip()]
    text = '\n'.join(cleaned_lines)
    
    # Remove multiple spaces
    text = ' '.join(text.split())
    
    # Remove very long sequences of dashes or equals
    import re
    text = re.sub(r'[-=]{10,}', '---', text)
    
    return text

# =============================================
# MAIN: Process All PDFs
# =============================================
def main():
    print("\n" + "="*60)
    print("📄 EXTRACTING TEXT FROM YOUR FARMING PDFs")
    print("="*60 + "\n")
    
    # Create folders if they don't exist
    os.makedirs(PDF_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Your specific PDF files
    target_pdfs = ["Farming Scemed.pdf", "farmerbook.pdf"]
    
    print(f"Looking for PDFs in: {PDF_FOLDER}")
    print(f"Target files: {', '.join(target_pdfs)}\n")
    
    # Check if PDFs exist
    found_pdfs = []
    missing_pdfs = []
    
    for pdf_name in target_pdfs:
        pdf_path = Path(PDF_FOLDER) / pdf_name
        if pdf_path.exists():
            found_pdfs.append(pdf_path)
            print(f"✅ Found: {pdf_name}")
        else:
            missing_pdfs.append(pdf_name)
            print(f"❌ Missing: {pdf_name}")
    
    if missing_pdfs:
        print(f"\n⚠️  Missing PDFs: {', '.join(missing_pdfs)}")
        print(f"\n💡 Please place these files in: {PDF_FOLDER}")
        print(f"   Current location should be:")
        print(f"   {os.path.abspath(PDF_FOLDER)}")
        
        if not found_pdfs:
            print(f"\n❌ No PDFs found. Cannot proceed.")
            return
        else:
            print(f"\n⚠️  Will process {len(found_pdfs)} available PDF(s)")
    
    print(f"\n{'='*60}")
    print(f"Processing {len(found_pdfs)} PDF file(s)...")
    print(f"{'='*60}\n")
    
    all_text = ""
    extracted_count = 0
    
    # Process each PDF
    for pdf_file in found_pdfs:
        print(f"📄 Processing: {pdf_file.name}")
        print(f"   File size: {pdf_file.stat().st_size / 1024:.1f} KB")
        
        # Extract text
        text = extract_text_from_pdf(pdf_file)
        
        if text:
            # Clean text
            cleaned_text = clean_text(text)
            
            print(f"   ✅ Text extracted successfully")
            print(f"   📊 Raw characters: {len(text):,}")
            print(f"   📊 Cleaned characters: {len(cleaned_text):,}")
            
            # Add document separator
            document_text = f"\n\n{'='*60}\n"
            document_text += f"DOCUMENT: {pdf_file.stem}\n"
            document_text += f"SOURCE FILE: {pdf_file.name}\n"
            document_text += f"{'='*60}\n\n"
            document_text += cleaned_text
            
            # Save individual file
            individual_output = os.path.join(OUTPUT_FOLDER, f"{pdf_file.stem}.txt")
            try:
                with open(individual_output, 'w', encoding='utf-8') as f:
                    f.write(cleaned_text)
                print(f"   ✅ Saved individual file: {individual_output}")
            except Exception as e:
                print(f"   ⚠️  Could not save individual file: {e}")
            
            # Add to combined text
            all_text += document_text
            extracted_count += 1
            print()
        else:
            print(f"   ❌ No text extracted from {pdf_file.name}")
            print(f"   💡 This might be a scanned image PDF requiring OCR\n")
    
    # Save combined file
    if all_text:
        try:
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                f.write(all_text)
            
            print("\n" + "="*60)
            print("✅ TEXT EXTRACTION COMPLETE!")
            print("="*60)
            print(f"\n📊 Statistics:")
            print(f"   - PDFs found: {len(found_pdfs)}")
            print(f"   - PDFs processed successfully: {extracted_count}")
            print(f"   - Total characters extracted: {len(all_text):,}")
            print(f"   - Combined file: {OUTPUT_FILE}")
            
            # Show file size
            file_size = os.path.getsize(OUTPUT_FILE) / 1024
            print(f"   - Output file size: {file_size:.1f} KB")
            
            print(f"\n📁 Individual files saved in: {OUTPUT_FOLDER}")
            
            # List individual files
            txt_files = list(Path(OUTPUT_FOLDER).glob("*.txt"))
            if txt_files:
                print(f"\n📄 Individual text files created:")
                for txt_file in txt_files:
                    if txt_file.name != "schemes_text.txt":
                        size = txt_file.stat().st_size / 1024
                        print(f"   - {txt_file.name} ({size:.1f} KB)")
            
            print(f"\n🚀 Next step: Build vector database")
            print(f"   cd scripts")
            print(f"   python build_vector_db.py")
            print("="*60 + "\n")
            
        except Exception as e:
            print(f"\n❌ Error saving combined file: {e}\n")
    else:
        print("\n" + "="*60)
        print("❌ NO TEXT EXTRACTED")
        print("="*60)
        print("\n💡 Possible issues:")
        print("   1. PDFs are scanned images (need OCR)")
        print("   2. PDFs are password protected")
        print("   3. PDFs are corrupted")
        print("\n💡 Solutions:")
        print("   1. Try using pdfplumber instead:")
        print("      pip install pdfplumber")
        print("   2. Use OCR tools like pytesseract")
        print("   3. Manually copy text from PDFs")
        print("="*60 + "\n")

# =============================================
# ALTERNATIVE: Using pdfplumber (if PyPDF2 fails)
# =============================================
def extract_with_pdfplumber(pdf_path):
    """
    Alternative extraction using pdfplumber
    Install with: pip install pdfplumber
    """
    try:
        import pdfplumber
        
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            print(f"   Using pdfplumber (better for complex PDFs)")
            for i, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
                    print(f"   ✓ Extracted page {i}/{len(pdf.pages)}")
        
        return text
    except ImportError:
        print("   ⚠️  pdfplumber not installed")
        print("   Install with: pip install pdfplumber")
        return None
    except Exception as e:
        print(f"   ❌ pdfplumber error: {e}")
        return None

# =============================================
# RUN
# =============================================
if __name__ == "__main__":
    main()
    
    # Instructions for missing PDFs
    print("\n💡 REMINDER: Place your PDFs in the correct location:")
    print(f"   {os.path.abspath(PDF_FOLDER)}")
    print("\n   Required files:")
    print("   - Farming Scemed.pdf")
    print("   - farmerbook.pdf")
    print("\n   After placing files, run this script again!")
    print()