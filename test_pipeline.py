try:
    from transformers import pipeline
    print("Pipeline imported successfully!")
except Exception as e:
    print(f"Error importing pipeline: {e}")
