import streamlit as st
import textstat
import nltk
from nltk.tokenize import sent_tokenize
import re
from typing import Dict, List, Tuple
from bs4 import BeautifulSoup
import uuid
import hashlib
from collections import Counter

# Ensure NLTK punkt data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Synonym dictionary for rephrasing
SYNONYMS = {
    'utilize': 'use',
    'enhance': 'improve',
    'advanced': 'modern',
    'performance': 'results',
    'technical': 'specialized',
    'evaluate': 'assess',
    'complex': 'complicated',
    'optimal': 'best',
    'demonstrate': 'show',
    'significant': 'important'
}

# Custom CSS for professional UI/UX
def load_css():
    st.markdown("""
        <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f4f7fa;
        }
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        h1, h2, h3 {
            color: #1a3c6d;
        }
        .stTextInput, .stTextArea, .stFileUploader {
            border-radius: 5px;
            border: 1px solid #d1d5db;
            padding: 10px;
        }
        .stButton>button {
            background-color: #1a3c6d;
            color: white;
            border-radius: 5px;
            padding: 10px 20px;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #2a5b9e;
        }
        .report-box {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-top: 20px;
        }
        .score {
            font-size: 24px;
            font-weight: bold;
            color: #28a745;
        }
        .feedback {
            color: #dc3545;
        }
        </style>
    """, unsafe_allow_html=True)

def parse_html_content(html_content: str, main_keyword: str) -> Tuple[str, str, List[str], str]:
    """Parse HTML or plain text content to extract title, H1, headers (H2/H3), and main content."""
    # Detect if input is plain text (no HTML tags)
    is_plain_text = not re.search(r'<[a-zA-Z]+>', html_content)
    
    if is_plain_text:
        # Convert plain text to HTML
        lines = html_content.strip().split('\n')
        html_lines = []
        headers = []
        content_parts = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith('# '):
                html_lines.append(f'<h1>{line[2:].strip()}</h1>')
            elif line.startswith('## '):
                html_lines.append(f'<h2>{line[3:].strip()}</h2>')
                headers.append(line[3:].strip())
            elif line.startswith('### '):
                html_lines.append(f'<h3>{line[4:].strip()}</h3>')
                headers.append(line[4:].strip())
            else:
                html_lines.append(f'<p>{line}</p>')
                content_parts.append(line)
        content = " ".join(content_parts)
        html_content = f"""
        <html>
            <head>
                <title>{main_keyword.capitalize()}</title>
            </head>
            <body>
                {''.join(html_lines)}
            </body>
        </html>
        """
    else:
        # Clean up Word/Google Docs artifacts
        html_content = re.sub(r'<o:p>.*?</o:p>', '', html_content)  # Remove Word's <o:p>
        html_content = re.sub(r'<span.*?>(.*?)</span>', r'\1', html_content)  # Remove Google Docs spans
        html_content = re.sub(r'<font.*?>(.*?)</font>', r'\1', html_content)  # Remove font tags
        html_content = re.sub(r'\s+', ' ', html_content)  # Normalize whitespace

        # Ensure HTML structure
        if not re.search(r'<html\b', html_content, re.IGNORECASE):
            html_content = f"""
            <html>
                <head>
                    <title>{main_keyword.capitalize()}</title>
                </head>
                <body>
                    {html_content}
                </body>
            </html>
            """

    soup = BeautifulSoup(html_content, 'html.parser')

    # Extract title
    title_tag = soup.find('title')
    title = title_tag.get_text().strip() if title_tag else main_keyword.capitalize()

    # Extract H1
    h1_tag = soup.find('h1')
    h1 = h1_tag.get_text().strip() if h1_tag else ""

    # Extract headers (H2, H3)
    headers = []
    for tag in soup.find_all(['h2', 'h3']):
        header_text = tag.get_text().strip()
        if header_text:
            headers.append(header_text)

    # Extract main content (text from p, div, article, etc.)
    content_tags = soup.find_all(['p', 'div', 'article', 'section'])
    content_parts = []
    for tag in content_tags:
        # Exclude tags that are headers or contain headers
        if not tag.find_all(['h1', 'h2', 'h3']):
            text = tag.get_text(separator=" ").strip()
            if text:
                content_parts.append(text)
    
    # Join content parts and clean up
    content = " ".join(content_parts).strip()
    content = re.sub(r'\s+', ' ', content)

    return title, h1, headers, content

def suggest_keywords(main_keyword: str, secondary_keywords: List[str], content: str, headers: List[str]) -> List[str]:
    """Suggest semantically related keywords based on main keyword and content."""
    words = content.lower().split() + " ".join(headers).lower().split()
    word_counts = Counter(words)
    
    # Generate suggestions based on main keyword and frequent content words
    main_words = set(main_keyword.lower().split())
    suggestions = []
    for word, count in word_counts.most_common(50):
        if word not in main_keyword.lower() and word not in secondary_keywords and len(word) > 3:
            if any(w in word or word in w for w in main_words):
                suggestions.append(word)
    
    # Combine with related phrases
    related_phrases = []
    for kw in [main_keyword] + secondary_keywords:
        for word in suggestions[:5]:
            related_phrases.append(f"{kw} {word}")
            related_phrases.append(f"{word} {kw}")
    
    return list(set(suggestions + related_phrases))[:5]

def analyze_tone(content: str) -> Tuple[float, str]:
    """Analyze tone consistency (academic/technical focus)."""
    sentences = sent_tokenize(content)
    academic_words = ['analysis', 'evaluate', 'assess', 'research', 'study', 'technical', 'methodology', 'data', 'results', 'significant']
    conversational_words = ['great', 'easy', 'awesome', 'cool', 'fun', 'simple', 'nice']
    
    academic_count = sum(content.lower().count(word) for word in academic_words)
    conversational_count = sum(content.lower().count(word) for word in conversational_words)
    total_sentences = len(sentences)
    
    academic_ratio = academic_count / total_sentences if total_sentences > 0 else 0
    conversational_ratio = conversational_count / total_sentences if total_sentences > 0 else 0
    
    if academic_ratio > conversational_ratio and academic_ratio > 0.1:
        score = 100 if academic_ratio > 0.2 else 70
        feedback = "" if score == 100 else "Tone mostly academic; increase use of formal terms like 'analysis' or 'evaluate'."
    elif conversational_ratio > academic_ratio and conversational_ratio > 0.1:
        score = 30
        feedback = "Tone too conversational; use formal terms like 'research' or 'technical' for academic style."
    else:
        score = 50
        feedback = "Inconsistent tone; maintain academic style with formal terms."
    
    return score, feedback

def check_plagiarism(content: str) -> Tuple[float, str]:
    """Basic plagiarism check using phrase repetition."""
    sentences = sent_tokenize(content)
    phrase_length = 5
    phrases = []
    for sent in sentences:
        words = sent.split()
        for i in range(len(words) - phrase_length + 1):
            phrase = " ".join(words[i:i+phrase_length])
            phrases.append(phrase)
    
    phrase_counts = Counter(phrases)
    repeated_phrases = sum(count - 1 for count in phrase_counts.values() if count > 1)
    total_phrases = len(phrases)
    repetition_ratio = repeated_phrases / total_phrases if total_phrases > 0 else 0
    
    if repetition_ratio > 0.1:
        score = 50
        feedback = "Content contains repeated phrases; ensure originality to avoid duplication."
    else:
        score = 100
        feedback = ""
    
    return score, feedback

def rephrase_sentence(sentence: str) -> str:
    """Rephrase a sentence using synonym replacement."""
    words = sentence.split()
    new_words = []
    for word in words:
        new_word = SYNONYMS.get(word.lower(), word)
        new_words.append(new_word if word.islower() else new_word.capitalize())
    return " ".join(new_words)

class ContentRater:
    """Class to rate blog content for SEO and readability before publishing."""
    
    def __init__(self, title: str, h1: str, headers: List[str], content: str, 
                 internal_links: List[str], main_keyword: str, secondary_keywords: List[str]):
        """Initialize with content details."""
        self.title = title.strip()
        self.h1 = h1.strip()
        self.headers = [h.strip() for h in headers]
        self.content = content.strip()
        self.internal_links = [link.strip() for link in internal_links]
        self.main_keyword = main_keyword.lower().strip()
        self.secondary_keywords = [kw.lower().strip() for kw in secondary_keywords]
        self.score = 0
        self.feedback = []
        self.keyword_suggestions = []

    def evaluate_seo(self) -> Dict[str, float]:
        """Evaluate SEO factors and return scores."""
        scores = {
            'title_tag': 0.0,
            'h1_tag': 0.0,
            'keyword_usage': 0.0,
            'content_depth': 0.0,
            'heading_structure': 0.0,
            'internal_links': 0.0
        }

        # Title tag (2025: front-load keyword, 60-70 chars)
        if self.title:
            if len(self.title) <= 70:
                scores['title_tag'] += 50
            if self.main_keyword in self.title.lower():
                scores['title_tag'] += 50
                if self.title.lower().startswith(self.main_keyword):
                    scores['title_tag'] += 20
            if scores['title_tag'] < 50:
                self.feedback.append("Title tag missing main keyword or too long (>70 chars).")

        # H1 tag (2025: single, descriptive, keyword-rich, concise, intent-aligned, natural)
        if self.h1:
            scores['h1_tag'] += 10
            if len(self.h1.split()) > 3:
                scores['h1_tag'] += 20
            else:
                self.feedback.append("H1 tag too short; make it descriptive (>3 words).")
            if self.main_keyword in self.h1.lower():
                scores['h1_tag'] += 30
            else:
                self.feedback.append("H1 tag missing main keyword.")
            intent_words = ['how', 'what', 'why', 'best', 'top', 'guide', 'buy', 'review']
            if any(word in self.h1.lower() for word in intent_words):
                scores['h1_tag'] += 10
            else:
                self.feedback.append("H1 tag may not align with search intent; include words like 'how', 'best', or 'guide'.")
            if len(self.h1) <= 60:
                scores['h1_tag'] += 20
            elif len(self.h1) <= 70:
                scores['h1_tag'] += 10
                self.feedback.append("H1 tag slightly long; aim for <60 chars.")
            else:
                self.feedback.append("H1 tag too long (>70 chars); keep it concise.")
            keyword_count = self.h1.lower().count(self.main_keyword)
            if keyword_count <= 2:
                scores['h1_tag'] += 10
            else:
                self.feedback.append("H1 tag may be keyword-stuffed; use keyword once or twice.")
            self.feedback.append("Ensure H1 is unique across pages and placed prominently as the main headline.")
        else:
            self.feedback.append("H1 tag missing; include a descriptive, keyword-rich H1.")

        # Keyword usage (2025: primary ≤2%, secondary presence, strategic placement)
        content_lower = self.content.lower()
        words = content_lower.split()
        word_count = len(words)

        # Count main keyword occurrences using regex for exact matches
        main_keyword_pattern = r'\b' + re.escape(self.main_keyword) + r'\b'
        main_keyword_count = len(re.findall(main_keyword_pattern, content_lower, re.IGNORECASE))
        main_density = (main_keyword_count / word_count) * 100 if word_count > 0 else 0

        # Count secondary keyword occurrences
        secondary_keyword_count = 0
        for kw in self.secondary_keywords:
            kw_pattern = r'\b' + re.escape(kw) + r'\b'
            secondary_keyword_count += len(re.findall(kw_pattern, content_lower, re.IGNORECASE))
        secondary_density = (secondary_keyword_count / word_count) * 100 if word_count > 0 else 0
        secondary_relative_density = (secondary_keyword_count / main_keyword_count) * 100 if main_keyword_count > 0 else float('inf')

        # Primary keyword evaluation
        if main_keyword_count > 0:
            # Placement
            first_100_words = " ".join(words[:100]).lower()
            last_100_words = " ".join(words[-100:]).lower() if len(words) > 100 else content_lower
            headers_text = " ".join(self.headers).lower()
            if re.search(main_keyword_pattern, self.h1.lower(), re.IGNORECASE):
                scores['keyword_usage'] += 15
            if re.search(main_keyword_pattern, first_100_words, re.IGNORECASE):
                scores['keyword_usage'] += 15
            if re.search(main_keyword_pattern, last_100_words, re.IGNORECASE):
                scores['keyword_usage'] += 15
            if re.search(main_keyword_pattern, headers_text, re.IGNORECASE):
                scores['keyword_usage'] += 10
            # Density
            if main_density <= 2.0:
                scores['keyword_usage'] += 20
            elif main_density <= 3.0:
                scores['keyword_usage'] += 10
                self.feedback.append("Primary keyword density slightly high (2-3%); aim for ≤2%.")
            else:
                scores['keyword_usage'] += 5
                self.feedback.append("Primary keyword density too high (>3%); possible stuffing.")
            # Natural integration (conversational/long-tail)
            if len(self.main_keyword.split()) > 2 or any(w in self.main_keyword for w in ['how', 'what', 'why']):
                scores['keyword_usage'] += 10
            else:
                self.feedback.append("Consider using a conversational or long-tail primary keyword (e.g., 'how to...').")
        else:
            self.feedback.append("Primary keyword not found in content.")

        # Secondary keyword evaluation
        if secondary_keyword_count > 0:
            # Placement
            headers_text = " ".join(self.headers).lower()
            links_text = " ".join(self.internal_links).lower()
            secondary_in_headers = any(re.search(r'\b' + re.escape(kw) + r'\b', headers_text, re.IGNORECASE) for kw in self.secondary_keywords)
            secondary_in_links = any(re.search(r'\b' + re.escape(kw) + r'\b', links_text, re.IGNORECASE) for kw in self.secondary_keywords)
            if secondary_in_headers:
                scores['keyword_usage'] += 10
            else:
                self.feedback.append("Include secondary keywords in H2/H3 subheadings.")
            if secondary_in_links:
                scores['keyword_usage'] += 5
            # Quantity
            expected_secondary = 2 if word_count < 1500 else 3
            if len(self.secondary_keywords) >= expected_secondary:
                scores['keyword_usage'] += 10
            else:
                self.feedback.append(f"Use at least {expected_secondary} secondary keywords for {word_count} words.")
            # Density
            if secondary_relative_density <= 100:
                scores['keyword_usage'] += 5
            else:
                self.feedback.append("Secondary keyword usage too high (>100% of primary); aim for balanced use.")
            # Semantic relevance
            primary_words = set(self.main_keyword.split())
            secondary_relevant = any(any(w in kw.split() for w in primary_words) for kw in self.secondary_keywords)
            if secondary_relevant:
                scores['keyword_usage'] += 5
            else:
                self.feedback.append("Secondary keywords may not align with primary keyword; ensure semantic relevance.")
        else:
            self.feedback.append("Secondary keywords not found in content.")

        # Penalties
        if main_density > 3.0 or secondary_relative_density > 100:
            scores['keyword_usage'] = max(0, scores['keyword_usage'] - 5)
        if secondary_keyword_count > 0 and not secondary_relevant:
            scores['keyword_usage'] = max(0, scores['keyword_usage'] - 3)

        self.feedback.append("Include primary keyword in URL slug and meta description.")

        # Content depth (2025: 1500+ words)
        if word_count >= 1500:
            scores['content_depth'] = 100
        elif word_count >= 800:
            scores['content_depth'] = 70
            self.feedback.append("Content is decent but aim for 1500+ words.")
        else:
            scores['content_depth'] = 30
            self.feedback.append("Content too short (<800 words).")

        # Heading structure (2025: multiple H2/H3 for skimmability)
        h2_h3_count = len(self.headers)
        if h2_h3_count >= 3:
            scores['heading_structure'] = 100
        elif h2_h3_count >= 1:
            scores['heading_structure'] = 60
            self.feedback.append("Add more H2/H3 subheadings for better structure.")
        else:
            scores['heading_structure'] = 30
            self.feedback.append("Missing H2/H3 tags; ensure multiple subheadings.")

        # Internal links (2025: 2-5 links per 1000 words)
        links_per_1000 = (len(self.internal_links) / (word_count / 1000)) if word_count > 0 else 0
        if 2 <= links_per_1000 <= 5:
            scores['internal_links'] = 100
        elif links_per_1000 > 0:
            scores['internal_links'] = 60
            self.feedback.append("Adjust internal links (2-5 per 1000 words).")
        else:
            scores['internal_links'] = 30
            self.feedback.append("No internal links; include 2-5 per 1000 words.")

        return scores

    def evaluate_readability(self) -> Dict[str, float]:
        """Evaluate readability factors and return scores."""
        scores = {
            'flesch_score': 0.0,
            'sentence_length': 0.0,
            'subheading_density': 0.0,
            'tone_consistency': 0.0,
            'plagiarism_check': 0.0
        }

        # Flesch-Kincaid score (2025: 30-50 for academic/technical audience)
        word_count = len(self.content.split())
        if word_count < 50:
            scores['flesch_score'] = 0
            self.feedback.append("Content too short for reliable readability scoring.")
        else:
            try:
                flesch_score = textstat.flesch_reading_ease(self.content)
                if 30 <= flesch_score <= 50:
                    scores['flesch_score'] = 100
                elif 10 <= flesch_score < 30 or 50 < flesch_score <= 60:
                    scores['flesch_score'] = 70
                    self.feedback.append("Flesch score slightly off; aim for 30-50 for academic/technical content.")
                else:
                    scores['flesch_score'] = 30
                    self.feedback.append("Flesch score too low/high; adjust for academic/technical audience (30-50).")
            except Exception as e:
                scores['flesch_score'] = 0
                self.feedback.append(f"Error calculating Flesch score: {str(e)}")

        # Sentence length (2025: <25% sentences >25 words for academic/technical)
        try:
            sentences = sent_tokenize(self.content)
            if len(sentences) < 5:
                scores['sentence_length'] = 0
                self.feedback.append("Too few sentences for reliable analysis.")
            else:
                long_sentences = sum(1 for sent in sentences if len(sent.split()) > 25)
                long_sentence_ratio = (long_sentences / len(sentences)) * 100
                if long_sentence_ratio <= 25:
                    scores['sentence_length'] = 100
                elif long_sentence_ratio <= 40:
                    scores['sentence_length'] = 70
                    self.feedback.append("Too many long sentences; keep most under 25 words.")
                else:
                    scores['sentence_length'] = 30
                    self.feedback.append("Excessive long sentences; simplify for readability.")
                
                # Rephrase long or complex sentences
                for sent in sentences:
                    if len(sent.split()) > 25 or textstat.flesch_reading_ease(sent) < 30:
                        rephrased = rephrase_sentence(sent)
                        if rephrased != sent:
                            self.feedback.append(f"Complex sentence: '{sent}' Rephrased: '{rephrased}'")
        except LookupError:
            scores['sentence_length'] = 0
            self.feedback.append("Sentence tokenization failed; ensure NLTK 'punkt' is installed.")
        except Exception as e:
            scores['sentence_length'] = 0
            self.feedback.append(f"Error analyzing sentence length: {str(e)}")

        # Subheading density (2025: 200-400 words per header, keyword mapping, readability)
        if word_count < 50:
            scores['subheading_density'] = 0
            self.feedback.append("Content too short for reliable subheading analysis.")
        elif not self.headers:
            scores['subheading_density'] = 0
            self.feedback.append("No H2/H3 subheadings; add at least one for structure.")
        else:
            words_per_subheading = word_count / len(self.headers)
            # Header-to-keyword mapping
            headers_with_keywords = 0
            for header in self.headers:
                header_lower = header.lower()
                keywords_in_header = sum(1 for kw in self.secondary_keywords if re.search(r'\b' + re.escape(kw) + r'\b', header_lower, re.IGNORECASE))
                if 1 <= keywords_in_header <= 2:
                    headers_with_keywords += 1
            keyword_mapping_score = 30 if headers_with_keywords >= len(self.headers) else 15 if headers_with_keywords > 0 else 0
            if keyword_mapping_score < 30:
                self.feedback.append("Assign 1-2 secondary keywords to each H2/H3 for topical depth.")

            # Semantic analysis
            primary_words = set(self.main_keyword.split())
            semantic_score = 20
            for header in self.headers:
                header_words = set(header.lower().split())
                if not any(word in header_words for word in primary_words):
                    semantic_score = 10
                    self.feedback.append(f"Header '{header}' may lack topical relevance; include main keyword terms.")
                    break

            # Readability audit
            readability_score = 20
            for header in self.headers:
                header_words = len(header.split())
                if header_words > 10:
                    self.feedback.append(f"Header '{header}' too long (>10 words); keep concise.")
                    readability_score = 10
                    break
                if not any(word in header.lower() for word in ['how', 'what', 'why', 'guide', 'tips', 'best', 'top']):
                    self.feedback.append(f"Header '{header}' lacks clarity; use descriptive/actionable terms.")
                    readability_score = 10
                    break

            # Words-per-subheading scoring
            density_score = 30
            if 200 <= words_per_subheading <= 400:
                density_score = 50
            elif 150 <= words_per_subheading < 200 or 400 < words_per_subheading <= 500:
                density_score = 30
                self.feedback.append("Adjust subheading frequency (1 per 200-400 words).")
            else:
                self.feedback.append("Poor subheading density; aim for 1 per 200-400 words.")

            scores['subheading_density'] = min(100, keyword_mapping_score + semantic_score + readability_score + density_score)

        # Tone consistency
        tone_score, tone_feedback = analyze_tone(self.content)
        scores['tone_consistency'] = tone_score
        if tone_feedback:
            self.feedback.append(tone_feedback)

        # Plagiarism check
        plagiarism_score, plagiarism_feedback = check_plagiarism(self.content)
        scores['plagiarism_check'] = plagiarism_score
        if plagiarism_feedback:
            self.feedback.append(plagiarism_feedback)

        return scores

    def calculate_final_score(self, seo_scores: Dict[str, float], readability_scores: Dict[str, float]) -> float:
        """Calculate final score (60% SEO, 40% Readability)."""
        seo_weight = 0.6
        readability_weight = 0.4
        seo_avg = sum(seo_scores.values()) / len(seo_scores)
        readability_avg = sum(readability_scores.values()) / len(readability_scores)
        self.score = (seo_avg * seo_weight) + (readability_avg * readability_weight)
        return self.score

    def generate_report(self) -> str:
        """Generate detailed report with scores and feedback."""
        self.keyword_suggestions = suggest_keywords(self.main_keyword, self.secondary_keywords, self.content, self.headers)
        seo_scores = self.evaluate_seo()
        readability_scores = self.evaluate_readability()
        final_score = self.calculate_final_score(seo_scores, readability_scores)

        report = [
            f"### Content Rating Report",
            f"**Main Keyword**: {self.main_keyword}",
            f"**Secondary Keywords**: {', '.join(self.secondary_keywords)}",
            f"**Final Score**: {final_score:.2f}/100",
            "",
            "**SEO Scores:**"
        ]
        for factor, score in seo_scores.items():
            report.append(f"- {factor.replace('_', ' ').title()}: {score:.2f}/100")
        report.append("\n**Readability Scores:**")
        for factor, score in readability_scores.items():
            report.append(f"- {factor.replace('_', ' ').title()}: {score:.2f}/100")
        report.append("\n**Feedback and Suggestions:**")
        for feedback in self.feedback:
            report.append(f"- {feedback}")
        report.append("\n**Improvement Tips:**")
        report.append("- Ensure title and H1 include main keyword and are <60 chars for H1, <70 for title.")
        report.append("- Include primary keyword in H1, first 100 words, conclusion, URL slug, and meta description.")
        report.append("- Use 2-3 secondary keywords per 1500 words in subheadings and links.")
        report.append(f"- Suggested keywords: {', '.join(self.keyword_suggestions)}")
        report.append("- Keep primary keyword density ≤2%, secondary balanced with primary usage.")
        report.append("- Use conversational or long-tail keywords for voice search.")
        report.append("- Ensure secondary keywords are semantically relevant to primary keyword.")
        report.append("- Write 1500+ words for in-depth content.")
        report.append("- Use multiple H2/H3 for structure (1 per 200-400 words).")
        report.append("- Assign 1-2 secondary keywords to each H2/H3 for topical depth.")
        report.append("- Ensure headers are concise (<10 words) and include descriptive terms.")
        report.append("- Add 2-5 internal links per 1000 words.")
        report.append("- Target Flesch score of 30-50 for academic/technical readability.")
        report.append("- Keep sentences under 25 words for clarity.")
        report.append("- Maintain academic tone with formal terms like 'analysis' or 'evaluate'.")
        report.append("- Ensure content originality to avoid plagiarism.")

        return "\n".join(report)

def main():
    """Streamlit app for content rating."""
    load_css()
    st.title("Pre-Publishing Content Rating Tool")
    st.markdown("Paste your HTML, Word, or Google Docs content below or upload an HTML file, and provide additional details to get an SEO and readability score.")
    st.markdown("""
        **Expected Formats:**
        - **HTML**: Full HTML with `<title>`, `<h1>`, `<h2>`, `<p>`, etc.
        - **Word/Google Docs**: Copy-pasted content with headings and paragraphs (will be converted to HTML).
        - **Plain Text**: Use `#` for H1, `##` for H2, `###` for H3, and new lines for paragraphs.
        **Example HTML:**
        ```html
        <html>
            <head>
                <title>Blog Title</title>
            </head>
            <body>
                <h1>H1 Tag</h1>
                <h2>H2 Header</h2>
                <p>Main content paragraphs...</p>
            </body>
        </html>
        ```
        **Example Plain Text:**
        ```
        # Running Shoe Innovations
        ## Running Footwear Materials
        Advanced running footwear utilizes carbon fiber...
        ### Athletic Shoes Cushioning
        Foam-based athletic shoes reduce impact...
        ```
    """)

    with st.form("content_form"):
        html_input = st.text_area("Paste Content (HTML, Word, Google Docs, or Plain Text)", placeholder="# Running Shoe Innovations\n## Running Footwear Materials\nAdvanced running footwear utilizes carbon fiber...\n### Athletic Shoes Cushioning\nFoam-based athletic shoes reduce impact...", height=300)
        uploaded_file = st.file_uploader("Or Upload HTML File", type=["html"])
        internal_links = st.text_area("Internal Links (one per line)", placeholder="/shoe-design\n/performance-tips")
        main_keyword = st.text_input("Main Keyword", placeholder="running shoe technology")
        secondary_keywords = st.text_input("Secondary Keywords (comma-separated)", placeholder="running footwear, athletic shoes")
        submit = st.form_submit_button("Rate Content")

    if submit:
        if not html_input and not uploaded_file:
            st.error("Please provide content by pasting it or uploading a file.")
        if not main_keyword:
            st.error("Please provide a Main Keyword.")
        else:
            try:
                # Get content from input or uploaded file
                if uploaded_file:
                    html_content = uploaded_file.read().decode("utf-8")
                else:
                    html_content = html_input

                # Parse the content
                title, h1, headers, content = parse_html_content(html_content, main_keyword)
                
                # Process other inputs
                internal_links_list = [link.strip() for link in internal_links.split("\n") if link.strip()] if internal_links else []
                secondary_keywords_list = [kw.strip() for kw in secondary_keywords.split(",") if kw.strip()] if secondary_keywords else []

                # Check if parsed components are present
                if not content:
                    st.error("No main content detected; ensure the input includes paragraphs or text.")
                else:
                    rater = ContentRater(
                        title=title,
                        h1=h1,
                        headers=headers,
                        content=content,
                        internal_links=internal_links_list,
                        main_keyword=main_keyword,
                        secondary_keywords=secondary_keywords_list
                    )
                    report = rater.generate_report()
                    st.markdown(f"<div class='report-box'><p class='score'>Final Score: {rater.score:.2f}/100</p>{report}</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error processing content: {str(e)}")
                st.markdown("Please ensure the input is valid (HTML, Word, Google Docs, or plain text) and try again.")

if __name__ == "__main__":
    main()
