import streamlit as st
import requests
from bs4 import BeautifulSoup
import validators
import urllib.parse
import json
from googleapiclient.discovery import build
import io
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
import os
import aiohttp
import asyncio
from collections import Counter
import re
from functools import lru_cache
import logging
from textstat import flesch_reading_ease, avg_sentence_length
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

# Download NLTK data
nltk.download('punkt', quiet=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load PageSpeed API Key from Streamlit secrets
PAGESPEED_API_KEY = os.getenv("PAGESPEED_API_KEY")

# Function to fetch page content
def get_page_content(url):
    try:
        logger.info(f"Fetching content for {url}")
        response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching {url}: {e}")
        return None

# Function to analyze metadata and score
def analyze_metadata(soup):
    metadata = {}
    score = 0
    feedback = []

    title = soup.title.string.strip() if soup.title else ""
    metadata["Title"] = title if title else "❌ No Title Found"
    title_length = len(title)
    if 50 <= title_length <= 60:
        score += 5
    elif title_length > 0:
        score += 2
        feedback.append("Title length should be 50–60 characters.")
    else:
        feedback.append("Missing title tag. Add a relevant title (50–60 characters).")

    meta_desc = soup.find("meta", attrs={"name": "description"})
    meta_desc_content = meta_desc["content"].strip() if meta_desc and meta_desc.get("content") else ""
    metadata["Meta Description"] = meta_desc_content if meta_desc_content else "❌ No Meta Description Found"
    desc_length = len(meta_desc_content)
    if 120 <= desc_length <= 160:
        score += 5
    elif desc_length > 0:
        score += 2
        feedback.append("Meta description should be 120–160 characters.")
    else:
        feedback.append("Missing meta description. Add one (120–160 characters).")

    return metadata, score, feedback

# Function to analyze headlines and score
def analyze_headlines(soup):
    headlines = {
        "H1": [h.get_text(strip=True) for h in soup.find_all("h1")],
        "H2": [h.get_text(strip=True) for h in soup.find_all("h2")],
        "H3": [h.get_text(strip=True) for h in soup.find_all("h3")]
    }
    score = 0
    feedback = []

    if len(headlines["H1"]) == 1:
        score += 5
    elif len(headlines["H1"]) == 0:
        feedback.append("Missing H1 tag. Add exactly one H1 with primary keyword.")
    else:
        score += 2
        feedback.append("Multiple H1 tags found. Use exactly one H1.")

    if len(headlines["H2"]) >= 2:
        score += 3
    elif len(headlines["H2"]) > 0:
        score += 1
        feedback.append("Add more H2 tags (at least 2) for better structure.")
    else:
        feedback.append("No H2 tags found. Add H2s for content hierarchy.")

    if len(headlines["H3"]) >= 1:
        score += 2
    else:
        feedback.append("Consider adding H3 tags for deeper content structure.")

    return headlines, score, feedback

# Asynchronous function to check a single link
async def check_link(session, url):
    try:
        async with session.head(url, timeout=5) as response:
            return url, response.status < 400
    except Exception as e:
        logger.error(f"Error checking link {url}: {e}")
        return url, False

# Function to analyze links and score
async def analyze_links(soup, base_url):
    links = soup.find_all("a", href=True)
    internal_links = []
    external_links = []
    broken_links = []

    async with aiohttp.ClientSession() as session:
        tasks = []
        for link in links[:50]:
            href = link["href"]
            full_url = urllib.parse.urljoin(base_url, href)
            if base_url in full_url:
                internal_links.append(full_url)
                tasks.append(check_link(session, full_url))
            else:
                external_links.append(full_url)
                tasks.append(check_link(session, full_url))

        results = await asyncio.gather(*tasks)
        broken_links = [url for url, is_valid in results if not is_valid]

    score = 0
    feedback = []
    if len(internal_links) >= 5:
        score += 5
    elif len(internal_links) > 0:
        score += 2
        feedback.append("Add more internal links (5+ recommended).")
    else:
        feedback.append("No internal links found. Add links to related pages.")

    if len(external_links) >= 1:
        score += 5
    else:
        feedback.append("No external links. Link to high-authority sites.")

    if len(broken_links) == 0:
        score += 5
    else:
        score += 2
        feedback.append(f"Found {len(broken_links)} broken links. Fix them.")

    return internal_links, external_links, broken_links, score, feedback

# Function to analyze content and score
def analyze_content(soup):
    text = soup.get_text(strip=True)
    tokens = word_tokenize(text.lower())
    fdist = FreqDist(tokens)
    word_count = len(tokens)
    score = 0
    feedback = []

    # Word Count
    if word_count >= 1000:
        score += 10
    elif word_count >= 300:
        score += 5
        feedback.append("Increase word count to 1000+ for better SEO.")
    else:
        feedback.append("Content too short. Aim for at least 300 words.")

    # Readability
    readability = flesch_reading_ease(text) if text else 0
    if readability >= 60:
        score += 5
    elif readability >= 30:
        score += 2
        feedback.append("Improve readability (Flesch score 60+).")
    else:
        feedback.append("Content is hard to read. Simplify language.")

    # Keyword Density
    keywords = extract_keywords(soup, top_n=1)
    primary_keyword = list(keywords.keys())[0] if keywords else ""
    keyword_count = keywords.get(primary_keyword, 0)
    density = (keyword_count / word_count * 100) if word_count > 0 else 0
    if 1 <= density <= 2:
        score += 5
    elif density > 0:
        score += 2
        feedback.append("Adjust primary keyword density to 1–2%.")
    else:
        feedback.append("No primary keyword detected. Optimize content.")

    avg_sentence_len = avg_sentence_length(text) if text else 0

    return word_count, readability, primary_keyword, density, avg_sentence_len, score, feedback

# Function to extract keywords
def extract_keywords(soup, top_n=5):
    text = soup.get_text(strip=True)
    words = word_tokenize(text.lower())
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    keywords = [word for word in words if word not in stop_words and len(word) > 3]
    keyword_counts = Counter(keywords)
    return dict(keyword_counts.most_common(top_n))

# Function to analyze images and alt texts
def analyze_images(soup):
    images = soup.find_all("img")
    image_data = []
    score = 0
    feedback = []

    for img in images:
        src = img.get("src", "❌ No Src")
        alt = img.get("alt", "❌ No Alt Text")
        image_data.append({"src": src, "alt": alt})

    if len(image_data) >= 3:
        score += 5
    elif len(image_data) > 0:
        score += 2
        feedback.append("Add more images (3+ recommended).")
    else:
        feedback.append("No images found. Add relevant images.")

    alt_missing = sum(1 for img in image_data if img["alt"] == "❌ No Alt Text")
    if alt_missing == 0 and len(image_data) > 0:
        score += 5
    elif alt_missing < len(image_data):
        score += 2
        feedback.append(f"{alt_missing} images missing alt text. Add descriptive alt texts.")
    else:
        feedback.append("All images lack alt text. Add alt texts for SEO.")

    return image_data, score, feedback

# Function to check technical SEO
def analyze_technical_seo(soup, url):
    metrics = {}
    score = 0
    feedback = []

    canonical = soup.find("link", rel="canonical")
    metrics["Canonical Tag"] = canonical["href"] if canonical and canonical.get("href") else "❌ No Canonical Tag"
    if canonical and canonical.get("href"):
        score += 5
    else:
        feedback.append("Missing canonical tag. Add one to specify the preferred URL.")

    robots = soup.find("meta", attrs={"name": "robots"})
    metrics["Robots Meta"] = robots["content"] if robots and robots.get("content") else "❌ No Robots Meta"
    if robots and "noindex" not in robots.get("content", "").lower():
        score += 3
    else:
        feedback.append("Robots meta missing or set to noindex. Ensure indexing is allowed.")

    scripts = soup.find_all("script", type="application/ld+json")
    metrics["Structured Data"] = "✅ Present" if scripts else "❌ Not Found"
    if scripts:
        score += 4
    else:
        feedback.append("No structured data found. Add Schema.org markup.")

    metrics["HTTPS"] = "✅ Secure" if url.startswith("https") else "❌ Not Secure"
    if url.startswith("https"):
        score += 3
    else:
        feedback.append("Use HTTPS for security and SEO.")

    return metrics, score, feedback

# Function to get Google PageSpeed Insights
@lru_cache(maxsize=100)
def get_pagespeed_insights(url, strategy="mobile"):
    try:
        if not PAGESPEED_API_KEY:
            return {
                "Performance Score": "⚠️ API Key Missing",
                "Core Web Vitals": {},
                "Mobile Friendliness": "N/A",
                "Error": "No valid API Key found. Set PAGESPEED_API_KEY in Streamlit secrets.",
                "Strategy": strategy.capitalize(),
                "SEO Score": 0,
                "Feedback": ["Set PAGESPEED_API_KEY in Streamlit secrets."]
            }

        logger.info(f"Fetching PageSpeed Insights for {url} ({strategy})")
        service = build("pagespeedonline", "v5", developerKey=PAGESPEED_API_KEY)
        result = service.pagespeedapi().runpagespeed(url=url, strategy=strategy).execute()

        lighthouse_data = result.get("lighthouseResult", {})
        categories = lighthouse_data.get("categories", {})
        audits = lighthouse_data.get("audits", {})

        performance_score = categories.get("performance", {}).get("score")
        performance_score = round(performance_score * 100) if performance_score is not None else 0

        core_web_vitals = {
            "First Contentful Paint (FCP)": audits.get("first-contentful-paint", {}).get("displayValue", "N/A"),
            "Largest Contentful Paint (LCP)": audits.get("largest-contentful-paint", {}).get("displayValue", "N/A"),
            "Cumulative Layout Shift (CLS)": audits.get("cumulative-layout-shift", {}).get("displayValue", "N/A"),
            "Total Blocking Time (TBT)": audits.get("total-blocking-time", {}).get("displayValue", "N/A"),
            "Speed Index": audits.get("speed-index", {}).get("displayValue", "N/A"),
        }

        mobile_friendly = audits.get("is-crawlable", {}).get("score", 0) == 1
        mobile_friendly = "✅ Mobile-Friendly" if mobile_friendly else "❌ Not Mobile-Friendly"

        score = 0
        feedback = []
        if performance_score >= 90:
            score += 10
        elif performance_score >= 50:
            score += 5
            feedback.append("Improve page speed (target 90+ for best results).")
        else:
            feedback.append("Page speed is low. Optimize assets and server response.")

        if mobile_friendly == "✅ Mobile-Friendly":
            score += 5
        else:
            feedback.append("Ensure the page is mobile-friendly.")

        cwv_pass = sum(1 for k, v in core_web_vitals.items() if v != "N/A" and "s" in v and float(v.split()[0]) < 2.5)
        if cwv_pass >= 3:
            score += 5
        else:
            feedback.append("Optimize Core Web Vitals (FCP, LCP, CLS).")

        return {
            "Performance Score": performance_score,
            "Core Web Vitals": core_web_vitals,
            "Mobile Friendliness": mobile_friendly,
            "Strategy": strategy.capitalize(),
            "SEO Score": score,
            "Feedback": feedback
        }

    except Exception as e:
        logger.error(f"PageSpeed API error for {url}: {e}")
        return {
            "Performance Score": "⚠️ Not Available",
            "Core Web Vitals": {},
            "Mobile Friendliness": "N/A",
            "Error": str(e),
            "Strategy": strategy.capitalize(),
            "SEO Score": 0,
            "Feedback": ["Failed to fetch PageSpeed data. Check API key and quota."]
        }

# Function to generate a PDF report
def generate_pdf_report(url, analysis, total_score):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("SEO Content Rating Report 2025", styles["Title"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph(f"Website: {url}", styles["Normal"]))
    story.append(Paragraph(f"Total SEO Score: {total_score}/100", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("🏷️ Metadata (10 points)", styles["Heading2"]))
    story.append(Paragraph(f"Score: {analysis['metadata_score']}/10", styles["Normal"]))
    story.append(Paragraph(f"Title: {analysis['metadata']['Title']}", styles["Normal"]))
    story.append(Paragraph(f"Meta Description: {analysis['metadata']['Meta Description']}", styles["Normal"]))
    for fb in analysis["metadata_feedback"]:
        story.append(Paragraph(f"• {fb}", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("📜 Headlines (10 points)", styles["Heading2"]))
    story.append(Paragraph(f"Score: {analysis['headlines_score']}/10", styles["Normal"]))
    for tag, headlines in analysis["headlines"].items():
        story.append(Paragraph(f"{tag} Count: {len(headlines)}", styles["Normal"]))
        for i, h in enumerate(headlines[:5], 1):
            story.append(Paragraph(f"{tag} #{i}: {h}", styles["Normal"]))
    for fb in analysis["headlines_feedback"]:
        story.append(Paragraph(f"• {fb}", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("📝 Content Quality (20 points)", styles["Heading2"]))
    story.append(Paragraph(f"Score: {analysis['content_score']}/20", styles["Normal"]))
    story.append(Paragraph(f"Word Count: {analysis['word_count']}", styles["Normal"]))
    story.append(Paragraph(f"Readability (Flesch): {analysis['readability']:.1f}", styles["Normal"]))
    story.append(Paragraph(f"Primary Keyword: {analysis['primary_keyword']}", styles["Normal"]))
    story.append(Paragraph(f"Keyword Density: {analysis['keyword_density']:.2f}%", styles["Normal"]))
    story.append(Paragraph(f"Average Sentence Length: {analysis['avg_sentence_len']:.1f} words", styles["Normal"]))
    for fb in analysis["content_feedback"]:
        story.append(Paragraph(f"• {fb}", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("🔗 Links (15 points)", styles["Heading2"]))
    story.append(Paragraph(f"Score: {analysis['links_score']}/15", styles["Normal"]))
    story.append(Paragraph(f"Internal Links: {len(analysis['internal_links'])}", styles["Normal"]))
    story.append(Paragraph(f"External Links: {len(analysis['external_links'])}", styles["Normal"]))
    story.append(Paragraph(f"Broken Links: {len(analysis['broken_links'])}", styles["Normal"]))
    for fb in analysis["links_feedback"]:
        story.append(Paragraph(f"• {fb}", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("🖼️ Images (10 points)", styles["Heading2"]))
    story.append(Paragraph(f"Score: {analysis['images_score']}/10", styles["Normal"]))
    story.append(Paragraph(f"Total Images: {len(analysis['images'])}", styles["Normal"]))
    for i, img in enumerate(analysis["images"][:5], 1):
        story.append(Paragraph(f"Image #{i} Src: {img['src']}", styles["Normal"]))
        story.append(Paragraph(f"Image #{i} Alt: {img['alt']}", styles["Normal"]))
    for fb in analysis["images_feedback"]:
        story.append(Paragraph(f"• {fb}", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("⚙️ Technical SEO (15 points)", styles["Heading2"]))
    story.append(Paragraph(f"Score: {analysis['technical_seo_score']}/15", styles["Normal"]))
    for metric, value in analysis["technical_seo"].items():
        story.append(Paragraph(f"{metric}: {value}", styles["Normal"]))
    for fb in analysis["technical_seo_feedback"]:
        story.append(Paragraph(f"• {fb}", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("⚡ PageSpeed Insights (20 points)", styles["Heading2"]))
    score = analysis["pagespeed"].get("SEO Score", 0)
    story.append(Paragraph(f"Score: {score}/20", styles["Normal"]))
    story.append(Paragraph(f"Mobile Performance Score: {analysis['pagespeed']['Performance Score']}/100", styles["Normal"]))
    story.append(Paragraph(f"Mobile Friendliness: {analysis['pagespeed']['Mobile Friendliness']}", styles["Normal"]))
    for metric, value in analysis["pagespeed"].get("Core Web Vitals", {}).items():
        story.append(Paragraph(f"{metric}: {value}", styles["Normal"]))
    for fb in analysis["pagespeed"].get("Feedback", []):
        story.append(Paragraph(f"• {fb}", styles["Normal"]))

    doc.build(story)
    buffer.seek(0)
    return buffer

# Streamlit UI
st.set_page_config(page_title="SEO Content Rater 2025", layout="wide")
st.title("🕵️‍♂️ SEO Content Rater 2025")
st.markdown("Rate your website's content based on **2025 SEO best practices**. Compare with a competitor and get actionable insights.")

url1 = st.text_input("🔗 Enter Your Website URL", "")
url2 = st.text_input("🆚 Enter Competitor Website URL (Optional)", "")

if st.button("🔍 Rate Content"):
    if not (validators.url(url1) and (not url2 or validators.url(url2))):
        st.error("❌ Please enter valid URLs.")
        st.stop()

    st.info("Analyzing content... Please wait.")
    progress_bar = st.progress(0)
    col1, col2 = st.columns(2) if url2 else (st, None)

    async def analyze_website(url, column, report_name):
        with column:
            st.subheader(f"Website: {url}")
            page_content = get_page_content(url)
            progress_bar.progress(25)
            if page_content:
                soup = BeautifulSoup(page_content, "html.parser")
                
                metadata, metadata_score, metadata_feedback = analyze_metadata(soup)
                headlines, headlines_score, headlines_feedback = analyze_headlines(soup)
                word_count, readability, primary_keyword, keyword_density, avg_sentence_len, content_score, content_feedback = analyze_content(soup)
                internal_links, external_links, broken_links, links_score, links_feedback = await analyze_links(soup, url)
                images, images_score, images_feedback = analyze_images(soup)
                technical_seo, technical_seo_score, technical_seo_feedback = analyze_technical_seo(soup, url)
                pagespeed = get_pagespeed_insights(url)

                analysis = {
                    "metadata": metadata,
                    "metadata_score": metadata_score,
                    "metadata_feedback": metadata_feedback,
                    "headlines": headlines,
                    "headlines_score": headlines_score,
                    "headlines_feedback": headlines_feedback,
                    "word_count": word_count,
                    "readability": readability,
                    "primary_keyword": primary_keyword,
                    "keyword_density": keyword_density,
                    "avg_sentence_len": avg_sentence_len,
                    "content_score": content_score,
                    "content_feedback": content_feedback,
                    "internal_links": internal_links,
                    "external_links": external_links,
                    "broken_links": broken_links,
                    "links_score": links_score,
                    "links_feedback": links_feedback,
                    "images": images,
                    "images_score": images_score,
                    "images_feedback": images_feedback,
                    "technical_seo": technical_seo,
                    "technical_seo_score": technical_seo_score,
                    "technical_seo_feedback": technical_seo_feedback,
                    "pagespeed": pagespeed
                }

                total_score = (
                    metadata_score +
                    headlines_score +
                    content_score +
                    links_score +
                    images_score +
                    technical_seo_score +
                    pagespeed.get("SEO Score", 0)
                )
                progress_bar.progress(75)

                st.write(f"**Total SEO Score**: {total_score}/100")
                st.write("**Metadata** (10 points)", {k: v for k, v in metadata.items()})
                st.write(f"Score: {metadata_score}/10", metadata_feedback)
                st.write("**Headlines** (10 points)", {k: v[:5] for k, v in headlines.items()})
                st.write(f"Score: {headlines_score}/10", headlines_feedback)
                st.write("**Content** (20 points)", {
                    "Word Count": word_count,
                    "Readability": f"{readability:.1f}",
                    "Primary Keyword": primary_keyword,
                    "Keyword Density": f"{keyword_density:.2f}%",
                    "Avg Sentence Length": f"{avg_sentence_len:.1f} words"
                })
                st.write(f"Score: {content_score}/20", content_feedback)
                st.write("**Links** (15 points)", {
                    "Internal": len(internal_links),
                    "External": len(external_links),
                    "Broken": len(broken_links)
                })
                st.write(f"Score: {links_score}/15", links_feedback)
                st.write("**Images** (10 points)", images[:5])
                st.write(f"Score: {images_score}/10", images_feedback)
                st.write("**Technical SEO** (15 points)", technical_seo)
                st.write(f"Score: {technical_seo_score}/15", technical_seo_feedback)
                st.write("**PageSpeed Insights** (20 points)", pagespeed)
                st.write(f"Score: {pagespeed.get('SEO Score', 0)}/20", pagespeed.get("Feedback", []))

                pdf_buffer = generate_pdf_report(url, analysis, total_score)
                st.download_button(
                    label="📄 Download Report",
                    data=pdf_buffer,
                    file_name=f"SEO_Content_Rating_{report_name}.pdf",
                    mime="application/pdf",
                )
            else:
                st.error("❌ Could not fetch the webpage.")
            progress_bar.progress(100)

    asyncio.run(analyze_website(url1, col1, "Primary"))

    if url2 and col2:
        progress_bar = st.progress(0)
        asyncio.run(analyze_website(url2, col2, "Competitor"))
