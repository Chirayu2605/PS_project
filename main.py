# ============================================
# main.py
# This is the ENTRY POINT of our project.
# Run this file to start everything:
#   python main.py
#
# It will:
#   1. Train the model
#   2. Show results (accuracy, confusion matrix)
#   3. Let you type your own reviews to test
# ============================================

from train import run_pipeline


# This runs only when you execute: python main.py
if __name__ == "__main__":

    # Run the full training and evaluation pipeline
    model, vocab, accuracy, cm = run_pipeline(
        csv_path="dataset.csv",
        learning_rate=0.01,
        n_epochs=15,
    )

    # ---- INTERACTIVE DEMO ----
    # After training, you can type your own reviews and see predictions
    print("\n" + "=" * 60)
    print(" INTERACTIVE DEMO -- type a review to classify")
    print(" (type 'quit' to exit)")
    print("=" * 60)

    # Import the functions we need for the demo
    from preprocessing import clean_text, tokenize
    from feature_extraction import text_to_vector
    import numpy as np

    while True:
        # Ask user to type a review
        review = input("\nEnter review: ").strip()

        # Check if user wants to quit
        if review.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        # Step 1: Clean the review
        cleaned = clean_text(review)

        # Step 2: Split into words
        tokens = tokenize(cleaned)

        # Step 3: Convert to number vector using our vocabulary
        vector = text_to_vector(tokens, vocab)

        # Step 4: Reshape to 2D (model expects rows of data)
        vector = vector.reshape(1, -1)

        # Step 5: Get prediction
        prediction = model.predict(vector)[0]

        # Step 6: Show result
        if prediction == 1:
            print("  -> Prediction: Positive (+)")
        else:
            print("  -> Prediction: Negative (-)")
