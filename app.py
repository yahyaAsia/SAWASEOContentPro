import streamlit as st
import textstat
import nltk
from nltk.tokenize import sent_tokenize
import re
from typing import Dict, List, Tuple
import uuid

# Ensure NLTK punkt data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

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
        .stTextInput, .stTextArea, .stNumberInput {
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

class ContentRater:
    """Class to rate blog content for SEO and readability before publishing."""
    
    def __init__(self, title: str, h1: str, headers: List[str], content: str, 
                 image_count: int, image_alt_texts: List[str], internal_links: List[str], 
                 main_keyword: str, secondary_keywords: List[str]):
        """Initialize with content details."""
        self.title = title.strip()
        self.h1 = h1.strip()
        self.headers = [h.strip() for h in headers]
        self.content = content.strip()
        self.image_count = image_count
        self.image_alt_texts = [alt.strip() for alt in image_alt_texts]
        self.internal_links = [link.strip() for link in internal_links]
        self.main_keyword = main_keyword.lower().strip()
        self.secondary_keywords = [kw.lower().strip() for kw in secondary_keywords]
        self.score = 0
        self.feedback = []

    def evaluate_seo(self) -> Dict[str, float]:
        """Evaluate SEO factors and return scores."""
        scores = {
            'title_tag': 0.0,
            'h1_tag': 0.0,
            'keyword_usage': 0.0,
            'content_depth': 0.0,
            'heading_structure': 0.0,
            'image_optimization': 0.0,
            'internal_links': 0.0,
            'eeat_signals': 0.0
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
            scores['h1_tag'] += 10  # Single H1
            if len(self.h1.split()) > 3:
                scores['h1_tag'] += 20  # Descriptive
            else:
                self.feedback.append("H1 tag too short; make it descriptive (>3 words).")
            if self.main_keyword in self.h1.lower():
                scores['h1_tag'] += 30  # Primary keyword
            else:
                self.feedback.append("H1 tag missing main keyword.")
            intent_words = ['how', 'what', 'why', 'best', 'top', 'guide', 'buy', 'review']
            if any(word in self.h1.lower() for word in intent_words):
                scores['h1_tag'] += 10  # Search intent
            else:
                self.feedback.append("H1 tag may not align with search intent; include words like 'how', 'best', or 'guide'.")
            if len(self.h1) <= 60:
                scores['h1_tag'] += 20  # Concise
            elif len(self.h1) <= 70:
                scores['h1_tag'] += 10
                self.feedback.append("H1 tag slightly long; aim for <60 chars.")
            else:
                self.feedback.append("H1 tag too long (>70 chars); keep it concise.")
            keyword_count = self.h1.lower().count(self.main_keyword)
            if keyword_count <= 2:
                scores['h1_tag'] += 10  # Natural
            else:
                self.feedback.append("H1 tag may be keyword-stuffed; use keyword once or twice.")
            self.feedback.append("Ensure H1 is unique across pages and placed prominently as the main headline.")
        else:
            self.feedback.append("H1 tag missing; include a descriptive, keyword-rich H1.")

        # Keyword usage (2025: primary ≤1.5%, secondary ≤0.5% of primary, strategic placement)
        words = self.content.lower().split()
        word_count = len(words)
        main_keyword_count = sum(1 for word in words if self.main_keyword in word)
        secondary_keyword_count = sum(1 for word in words for kw in self.secondary_keywords if kw in word)
        main_density = (main_keyword_count / word_count) * 100 if word_count > 0 else 0
        secondary_density = (secondary_keyword_count / word_count) * 100 if word_count > 0 else 0
        secondary_relative_density = (secondary_keyword_count / main_keyword_count) * 100 if main_keyword_count > 0 else float('inf')

        # Primary keyword evaluation
        if main_keyword_count > 0:
            # Placement
            first_100_words = " ".join(words[:100])
            last_100_words = " ".join(words[-100:]) if len(words) > 100 else " ".join(words)
            if self.main_keyword in self.h1.lower():
                scores['keyword_usage'] += 10
            if self.main_keyword in first_100_words:
                scores['keyword_usage'] += 10
            if self.main_keyword in last_100_words:
                scores['keyword_usage'] += 10
            # Density
            if main_density <= 1.5:
                scores['keyword_usage'] += 20
            elif 1.5 < main_density <= 2:
                scores['keyword_usage'] += 10
                self.feedback.append("Primary keyword density slightly high (1.5-2%); aim for ≤1.5%.")
            else:
                scores['keyword_usage'] += 5
                self.feedback.append("Primary keyword density too high (>2%); possible stuffing.")
            # Natural integration (conversational/long-tail)
            if len(self.main_keyword.split()) > 3 or any(w in self.main_keyword for w in ['how', 'what', 'why']):
                scores['keyword_usage'] += 10
            else:
                self.feedback.append("Use conversational or long-tail primary keyword (e.g., 'how to...').")
        else:
            self.feedback.append("Primary keyword not found in content.")

        # Secondary keyword evaluation
        secondary_relevant = False  # Initialize to avoid unbound variable
        if secondary_keyword_count > 0:
            # Placement
            headers_text = " ".join(self.headers).lower()
            alt_texts = " ".join(self.image_alt_texts).lower()
            links_text = " ".join(self.internal_links).lower()
            if any(kw in headers_text for kw in self.secondary_keywords):
                scores['keyword_usage'] += 10
            else:
                self.feedback.append("Include secondary keywords in H2/H3 subheadings.")
            if any(kw in alt_texts for kw in self.secondary_keywords):
                scores['keyword_usage'] += 5
            if any(kw in links_text for kw in self.secondary_keywords):
                scores['keyword_usage'] += 5
            # Quantity and density
            expected_secondary = 2 if word_count < 1500 else 3
            if len(self.secondary_keywords) >= expected_secondary:
                scores['keyword_usage'] += 10
            else:
                self.feedback.append(f"Use {expected_secondary} secondary keywords per {word_count} words.")
            if secondary_relative_density <= 50:
                scores['keyword_usage'] += 5
            else:
                self.feedback.append("Secondary keyword usage too high (>50% of primary); aim for ≤50%.")
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
        if main_density > 1.5 or secondary_relative_density > 50:
            scores['keyword_usage'] = max(0, scores['keyword_usage'] - 10)  # Stuffing penalty
        if not secondary_relevant and secondary_keyword_count > 0:
            scores['keyword_usage'] = max(0, scores['keyword_usage'] - 5)  # Irrelevant penalty

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

        # Image optimization (2025: alt text with keywords, 1-3 images per 1000 words)
        if self.image_count > 0:
            images_per_1000 = (self.image_count / (word_count / 1000)) if word_count > 0 else 0
            if 1 <= images_per_1000 <= 3:
                scores['image_optimization'] += 50
            else:
                scores['image_optimization'] += 20
                self.feedback.append("Adjust image count (1-3 per 1000 words).")
            if all(self.main_keyword in alt.lower() for alt in self.image_alt_texts if alt):
                scores['image_optimization'] += 50
            else:
                self.feedback.append("Ensure all image alt texts include main keyword.")
        else:
            self.feedback.append("No images provided; include 1-3 per 1000 words.")

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

        # E-E-A-T signals (2025: author mention, related keywords)
        if any(kw in self.content.lower() for kw in ['author', 'expert', 'source', 'citation']):
            scores['eeat_signals'] = 100
        else:
            scores['eeat_signals'] = 30
            self.feedback.append("Mention author or sources for E-E-A-T.")

        return scores

    def evaluate_readability(self) -> Dict[str, float]:
        """Evaluate readability factors and return scores."""
        scores = {
            'flesch_score': 0.0,
            'sentence_length': 0.0,
            'subheading_density': 0.0
        }

        # Flesch-Kincaid score (2025: 60-70 for general audience)
        try:
            flesch_score = textstat.flesch_reading_ease(self.content)
            if 60 <= flesch_score <= 70:
                scores['flesch_score'] = 100
            elif 50 <= flesch_score < 60 or 70 < flesch_score <= 80:
                scores['flesch_score'] = 70
                self.feedback.append("Flesch score slightly off (aim for 60-70).")
            else:
                scores['flesch_score'] = 30
                self.feedback.append("Flesch score too low/high; adjust for general audience.")
        except Exception as e:
            scores['flesch_score'] = 0
            self.feedback.append(f"Error calculating Flesch score: {str(e)}")

        # Sentence length (2025: <25% sentences >20 words)
        try:
            sentences = sent_tokenize(self.content)
            long_sentences = sum(1 for sent in sentences if len(sent.split()) > 20)
            long_sentence_ratio = (long_sentences / len(sentences)) * 100 if sentences else 100
            if long_sentence_ratio <= 25:
                scores['sentence_length'] = 100
            elif long_sentence_ratio <= 40:
                scores['sentence_length'] = 70
                self.feedback.append("Too many long sentences; keep most under 20 words.")
            else:
                scores['sentence_length'] = 30
                self.feedback.append("Excessive long sentences; simplify for readability.")
        except LookupError:
            scores['sentence_length'] = 0
            self.feedback.append("Sentence tokenization failed; ensure NLTK 'punkt' is installed.")
        except Exception as e:
            scores['sentence_length'] = 0
            self.feedback.append(f"Error analyzing sentence length: {str(e)}")

        # Subheading density (2025: 1 subheading per 250-300 words)
        word_count = len(self.content.split())
        subheading_count = len(self.headers)
        words_per_subheading = word_count / subheading_count if subheading_count > 0 else float('inf')
        if 250 <= words_per_subheading <= 300:
            scores['subheading_density'] = 100
        elif 200 <= words_per_subheading < 250 or 300 < words_per_subheading <= 400:
            scores['subheading_density'] = 70
            self.feedback.append("Adjust subheading frequency (1 per 250-300 words).")
        else:
            scores['subheading_density'] = 30
            self.feedback.append("Poor subheading density; aim for 1 per 250-300 words.")

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
        report.append("- Use 2-3 secondary keywords per 1500 words in subheadings, alt texts, and links.")
        report.append("- Keep primary keyword density ≤1.5%, secondary ≤50% of primary usage.")
        report.append("- Use conversational or long-tail keywords for voice search.")
        report.append("- Ensure secondary keywords are semantically relevant to primary keyword.")
        report.append("- Write 1500+ words for in-depth content.")
        report.append("- Use multiple H2/H3 for structure (1 per 250-300 words).")
        report.append("- Include 1-3 images per 1000 words with keyword-rich alt text.")
        report.append("- Add 2-5 internal links per 1000 words.")
        report.append("- Mention author or sources for E-E-A-T.")
        report.append("- Target Flesch score of 60-70 for readability.")
        report.append("- Keep sentences short (<20 words).")

        return "\n".join(report)

def main():
    """Streamlit app for content rating."""
    load_css()
    st.title("Pre-Publishing Content Rating Tool")
    st.markdown("Enter your blog content details to get an SEO and readability score.")

    with st.form("content_form"):
        title = st.text_input("Blog Title", placeholder="Best Running Shoes for 2025")
        h1 = st.text_input("H1 Tag", placeholder="Top Running Shoes Reviewed")
        headers = st.text_area("Headers (H2/H3, one per line)", placeholder="Why Choose Quality Shoes?\nTop Picks for 2025")
        content = st.text_area("Main Content", placeholder="Running shoes are essential for performance...", height=200)
        image_count = st.number_input("Number of Images", min_value=0, value=0)
        image_alt_texts = st.text_area("Image Alt Texts (one per line)", placeholder="Running shoes on trail\nBest athletic shoes")
        internal_links = st.text_area("Internal Links (one per line)", placeholder="/shoe-care\n/running-tips")
        main_keyword = st.text_input("Main Keyword", placeholder="best running shoes")
        secondary_keywords = st.text_area("Secondary Keywords (one per line)", placeholder="running footwear\nathletic shoes")
        submit = st.form_submit_button("Rate Content")

    if submit:
        if not all([title, h1, content, main_keyword]):
            st.error("Please fill in all required fields (Title, H1, Content, Main Keyword).")
        else:
            headers_list = [h for h in headers.split("\n") if h.strip()] if headers else []
            image_alt_list = [alt for alt in image_alt_texts.split("\n") if alt.strip()] if image_alt_texts else []
            internal_links_list = [link for link in internal_links.split("\n") if link.strip()] if internal_links else []
            secondary_keywords_list = [kw for kw in secondary_keywords.split("\n") if kw.strip()] if secondary_keywords else []

            try:
                rater = ContentRater(
                    title=title,
                    h1=h1,
                    headers=headers_list,
                    content=content,
                    image_count=image_count,
                    image_alt_texts=image_alt_list,
                    internal_links=internal_links_list,
                    main_keyword=main_keyword,
                    secondary_keywords=secondary_keywords_list
                )
                report = rater.generate_report()
                st.markdown(f"<div class='report-box'><p class='score'>Final Score: {rater.score:.2f}/100</p>{report}</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error processing content: {str(e)}")
                st.markdown("Please ensure all inputs are valid and try again.")

if __name__ == "__main__":
    main()
