# Troubleshooting: ModuleNotFoundError for langchain.chains

## The Problem

If you're getting an error like this when running your Streamlit app:

```
ModuleNotFoundError: No module named 'langchain.chains'
```

This is because **LangChain 0.3.0 and later versions removed the `langchain.chains` module** and restructured the library. Your code is trying to import:

```python
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
```

But this module no longer exists in newer versions of LangChain.

## The Solution

I've fixed this by **pinning the LangChain version to 0.2.x** in your `requirements.txt`:

```
langchain>=0.2.0,<0.3.0
```

This ensures compatibility with your existing code that uses `langchain.chains`.

## What Changed in LangChain 0.3.0+

The LangChain library was significantly restructured:
- The `langchain.chains` module was removed
- Functions like `create_history_aware_retriever` were moved to different packages
- Many sub-packages were reorganized

Using version 0.2.x ensures your current code will work without modifications.

## Alternative Solutions

If you need to use a newer version of LangChain (0.3.0+), you would need to:
1. Update all import statements to match the new structure
2. Modify your code to use the new API
3. Refer to the [LangChain migration guide](https://python.langchain.com/docs/migration/) for details

However, for now, sticking with LangChain 0.2.x is the easiest solution.

## Installing the Fix

After updating `requirements.txt`, make sure to reinstall the dependencies:

```bash
pip install -r requirements.txt
```

If you're on Streamlit Cloud or another cloud platform, the requirements.txt will be automatically used when you deploy.

