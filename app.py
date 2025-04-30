import streamlit as st
import textstat
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import nltk

# Download NLTK data (only needed the first time)
nltk.download('punkt')

# Streamlit app title
st.title("SEO Content Rating Tool")

# Sidebar for user inputs
st.sidebar.header("Input Content Details")
title = st.sidebar.text_input("Title", "")
meta_description = st.sidebar.text_area("Meta Description", "")
main_keyword = st.sidebar.text_input("Main Keyword", "")
secondary_keywords = st.sidebar.text_input("Secondary Keywords (comma-separated)", "")
blog_content = st.text_area("Blog Content", height=300)  # Main content input

# Analyze button
if st.button("Analyze Content"):
    # Validate inputs
    if not blog_content:
        st.error("Please enter blog content to analyze!")
    else:
        # Tokenize blog content
        tokens = word_tokenize(blog_content.lower())
        fdist = FreqDist(tokens)
        total_words = len(tokens)

        # Keyword density
        main_keyword_count = fdist[main_keyword.lower()]
        secondary_keywords_list = [kw.strip().lower() for kw in secondary_keywords.split(',')]
        secondary_keyword_count = sum(fdist[keyword] for keyword in secondary_keywords_list)

        # Readability analysis
        readability_score = textstat.flesch_reading_ease(blog_content)
        avg_sentence_length = textstat.avg_sentence_length(blog_content)

        # Header and image analysis
        h1_count = blog_content.lower().count('<h1>')
        h2_count = blog_content.lower().count('<h2>')
        img_count = blog_content.lower().count('<img')
        img_with_alt = blog_content.lower().count('alt="')

        # Internal and external links
        internal_links = blog_content.lower().count('href="/')
        external_links = blog_content.lower().count('href="http')

        # Display results
        st.subheader("SEO Analysis Results")
        st.write(f"**Title Length:** {len(title)} characters")
        st.write(f"**Meta Description Length:** {len(meta_description)} characters")
        st.write(f"**Word Count:** {total_words} words")
        st.write(f"**Main Keyword Density:** {main_keyword_count / total_words * 100:.2f}%")
        st.write(f"**Secondary Keywords Density:** {secondary_keyword_count / total_words * 100:.2f}%")
        st.write(f"**H1 Tags:** {h1_count}")
        st.write(f"**H2 Tags:** {h2_count}")
        st.write(f"**Images Total:** {img_count}")
        st.write(f"**Images with Alt Text:** {img_with_alt}")
        st.write(f"**Internal Links:** {internal_links}")
        st.write(f"**External Links:** {external_links}")

        st.subheader("Readability Analysis Results")
        st.write(f"**Flesch Reading Ease Score:** {readability_score:.2f}")
        st.write(f"**Average Sentence Length:** {avg_sentence_length:.2f} words")

        # Suggestions
        st.subheader("Suggestions for Improvement")
        if len(title) < 50 or len(title) > 60:
            st.warning("Your title length is not optimal. Aim for 50-60 characters.")
        if len(meta_description) < 50 or len(meta_description) > 160:
            st.warning("Your meta description length is not optimal. Aim for 50-160 characters.")
        if main_keyword_count / total_words < 1 or main_keyword_count / total_words > 2.5:
            st.warning("Your main keyword density is not optimal. Aim for 1-2.5%.")
        if readability_score < 60:
            st.warning("Your content readability is low. Aim for a higher Flesch Reading Ease score (>60).")
        if h1_count == 0:
            st.warning("No H1 tags found. Every page should have one H1 tag.")
        if img_count > 0 and img_with_alt < img_count:
            st.warning("Some images are missing alt text. Add alt attributes for better SEO.")
        if internal_links < 3:
            st.warning("Consider adding more internal links to improve site navigation.")
        if external_links < 2:
            st.warning("Consider adding external links to credible sources.")

        st.success("Analysis complete! Make improvements as suggested to optimize your content.")
