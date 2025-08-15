import streamlit as st
from utils import process_text, read_file

def main():
    st.title("Task Extraction and Categorization")

    st.markdown("## Input")
    st.write("Enter text manually or upload a text file to extract and categorize tasks.")

    input_method = st.radio("Choose input method", ("Manual Input", "Upload File"))

    raw_text = ""
    if input_method == "Manual Input":
        raw_text = st.text_area("Enter the text here", height=200)
    else:
        uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
        if uploaded_file is not None:
            raw_text = read_file(uploaded_file)

    if raw_text:
        st.markdown("## Processing...")
        categorized_tasks, topics = process_text(raw_text)

        st.markdown("### Extracted Tasks")
        if categorized_tasks:
            for idx, task in enumerate(categorized_tasks, 1):
                st.write(f"**Task {idx}:** {task['task']}")
                st.write(f"- **Person:** {task['person'] if task['person'] else 'Not Specified'}")
                st.write(f"- **Deadline:** {task['deadline'] if task['deadline'] else 'Not Specified'}")
                st.write(f"- **Category:** {task['category']}")
                st.markdown("---")
        else:
            st.write("No tasks were identified in the input text.")

        st.markdown("### LDA Topics from Tasks")
        if topics:
            for topic in topics:
                st.write(topic)
        else:
            st.write("No topics to display.")
    else:
        st.info("Please enter some text or upload a file to proceed.")

if __name__ == '__main__':
    main()
