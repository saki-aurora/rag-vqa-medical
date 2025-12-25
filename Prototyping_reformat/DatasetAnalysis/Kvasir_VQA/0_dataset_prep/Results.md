Analysed Results from the Dataset
====================================

### 1. The "Vitals" (Volume Stats)

The exact size of the dataset is 

* **Total Questions:** 58,849
* **Total Images:** 6,500
* **Density:** On average, there are **9 questions per image**, but it varies wildly (see below).


### 2. The "Diagnosis" (Biases Detected)

The script revealed three massive imbalances that would confuse a model if left unchecked:

**The "Sick vs. Healthy" Imbalance:**
* They have **2,500 images** of "Normal" anatomy, but only **2,500 questions** for them (1 question per image).
* They have **1,000 images** of "Ulcerative Colitis," but **16,890 questions** for them (~17 questions per image).
* *Result:* The model will learn *much* more about diseases than healthy tissue because the "sick" images are discussed 17x more often.


**The "Yes/No" Trap:**
* **26,515** questions (about 45% of the total) are **Yes/No** questions.
* If the model just flips a coin or learns the most common answer (e.g., "Yes"), it can look artificially smart without understanding the image.


**The "Repetitive Template" Problem:**
* The script found that thousands of questions are identical strings.
* *Top Question:* "Have all polyps been removed?" appears **3,945 times**.
* *Runner Up:* "Is this finding easy to detect?" appears **3,941 times**.
* *Result:* The model might memorize the answer to "Have all polyps..." instead of looking for polyps.



### 3. The "Hidden" Data (Answer Types)

They categorized what the AI is expected to output:

* **Yes/No:** ~15k answers.
* **None/NA:** ~13.5k answers. (This is huge—it means often the answer is "Nothing to see here" or "Not relevant").
* **Numeric:** ~10k answers (matches the "How many..." questions).

### Summary

If we handed this report to a Data Scientist, they would say:

> *"This dataset is heavily biased toward pathological cases. It relies too much on repetitive Yes/No questions, and 'Normal' images are under-annotated. We need to be careful not to train a model that just guesses 'Yes' or 'None'."*