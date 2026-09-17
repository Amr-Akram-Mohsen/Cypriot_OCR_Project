# metrics_evaluator.py
def calculate_cer(reference_text, predicted_text):
    """حساب نسبة الخطأ على مستوى الحرف (Character Error Rate)"""
    import Levenshtein
    distance = Levenshtein.distance(reference_text, predicted_text)
    if len(reference_text) == 0:
        return 0.0
    return (distance / len(reference_text)) * 100

def calculate_wer(reference_words, predicted_words):
    """حساب نسبة الخطأ على مستوى الكلمة (Word Error Rate)"""
    import Levenshtein
    # تحويل القوائم إلى كلمات ومقارنتها
    ref_str = " ".join(reference_words)
    pred_str = " ".join(predicted_words)
    distance = Levenshtein.distance(ref_str, pred_str)
    if len(reference_words) == 0:
        return 0.0
    return (distance / len(reference_words)) * 100

# مثال للاستخدام والتجربة:
ground_truth = "𐠀𐠁𐠂"
prediction = "𐠀𐠁𐠃"

cer_score = calculate_cer(ground_truth, prediction)
print(f"Character Error Rate (CER): {cer_score:.2f}%")