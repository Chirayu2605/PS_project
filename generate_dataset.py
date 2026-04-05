"""
Dataset Generator
=================
Creates a CSV dataset with 2000+ short reviews labeled as
positive (1) or negative (0).  The reviews are generated from
template patterns to ensure variety and clear sentiment signals.
"""

import csv
import random

random.seed(42)

# ── Building blocks ──────────────────────────────────────────────────

positive_templates = [
    "This product is {adj}! I {verb} it.",
    "Absolutely {adj}. {reason}.",
    "I am so {emotion} with this purchase. {extra}",
    "Best {noun} I have ever {verb_past}! Highly recommend.",
    "{adj} quality and {adj2} service.",
    "Loved every bit of it. {reason}.",
    "Five stars! {reason}.",
    "Would buy again. {adj} value for the price.",
    "The {noun} exceeded my expectations. So {adj}!",
    "Great experience. {reason}. Truly {adj}.",
    "This is exactly what I needed. {adj} and {adj2}.",
    "I was {emotion} by how {adj} this {noun} is.",
    "My family loves this {noun}. {reason}.",
    "Incredible {noun}. {adj} build and {adj2} design.",
    "Superb! The {noun} works {adv} well.",
    "Pleasantly surprised. {reason}.",
    "Such a {adj} {noun}. Makes my day better.",
    "Highly recommended! {adj} and {adj2}.",
    "I {verb} this {noun} every day. Totally {adj}.",
    "Top notch {noun}. {reason}.",
]

negative_templates = [
    "This product is {adj}. I {verb} it.",
    "Absolutely {adj}. {reason}.",
    "I am so {emotion} with this purchase. {extra}",
    "Worst {noun} I have ever {verb_past}. Do not recommend.",
    "{adj} quality and {adj2} service.",
    "Hated every bit of it. {reason}.",
    "One star. {reason}.",
    "Would never buy again. {adj} value for the price.",
    "The {noun} failed my expectations. So {adj}.",
    "Terrible experience. {reason}. Truly {adj}.",
    "This is not what I expected. {adj} and {adj2}.",
    "I was {emotion} by how {adj} this {noun} is.",
    "My family dislikes this {noun}. {reason}.",
    "Awful {noun}. {adj} build and {adj2} design.",
    "Very poor. The {noun} works {adv} badly.",
    "Extremely let down. {reason}.",
    "Such a {adj} {noun}. Ruined my day.",
    "Do not recommend. {adj} and {adj2}.",
    "I {verb} this {noun} once and returned it. Totally {adj}.",
    "Bottom tier {noun}. {reason}.",
]

# ── Word pools ───────────────────────────────────────────────────────

pos_adj    = ["amazing", "fantastic", "wonderful", "excellent", "brilliant",
              "outstanding", "superb", "delightful", "perfect", "great",
              "impressive", "stellar", "exceptional", "marvelous", "fabulous",
              "splendid", "remarkable", "glorious", "elegant", "lovely"]
neg_adj    = ["terrible", "awful", "horrible", "dreadful", "poor",
              "disappointing", "pathetic", "mediocre", "broken", "useless",
              "flimsy", "defective", "subpar", "lousy", "annoying",
              "frustrating", "atrocious", "abysmal", "cheap", "ugly"]

pos_adj2   = ["fast", "reliable", "smooth", "premium", "friendly",
              "polished", "clean", "intuitive", "responsive", "sturdy"]
neg_adj2   = ["slow", "unreliable", "clunky", "cheap", "rude",
              "confusing", "messy", "unintuitive", "laggy", "fragile"]

pos_emotion = ["happy", "thrilled", "pleased", "satisfied", "delighted",
               "grateful", "excited", "impressed", "overjoyed", "amazed"]
neg_emotion = ["disappointed", "frustrated", "annoyed", "upset", "angry",
               "disgusted", "unhappy", "irritated", "dissatisfied", "furious"]

pos_verb   = ["love", "enjoy", "adore", "appreciate", "cherish",
              "recommend", "admire", "value", "treasure", "praise"]
neg_verb   = ["hate", "regret", "despise", "dislike", "loathe",
              "returned", "avoid", "reject", "criticized", "resent"]

pos_verb_past = ["owned", "used", "tried", "purchased", "experienced"]
neg_verb_past = ["bought", "used", "tried", "wasted money on", "suffered through"]

nouns = ["product", "item", "phone", "laptop", "headphones", "camera",
         "speaker", "tablet", "watch", "charger", "keyboard", "mouse",
         "monitor", "printer", "blender", "toaster", "backpack", "jacket",
         "shoes", "book", "course", "service", "app", "software", "game"]

pos_reasons = [
    "The build quality is top notch",
    "It arrived on time and well packaged",
    "Customer service was very helpful",
    "Easy to set up and use right away",
    "Worth every penny I spent",
    "Battery life is incredible",
    "The design is sleek and modern",
    "Performance is smooth and fast",
    "I have recommended it to all my friends",
    "Exceeded all my expectations",
    "The materials feel premium",
    "Works perfectly out of the box",
    "Great attention to detail",
    "The colors are vibrant and beautiful",
    "Very comfortable to use for long hours",
]

neg_reasons = [
    "The build quality is very cheap",
    "It arrived damaged and late",
    "Customer service was unhelpful",
    "Took hours to set up and still does not work",
    "Total waste of money",
    "Battery dies within an hour",
    "The design looks outdated and bulky",
    "Performance is sluggish and buggy",
    "I have warned all my friends to avoid it",
    "Failed to meet even basic expectations",
    "The materials feel flimsy",
    "Was broken right out of the box",
    "No attention to detail at all",
    "The colors faded after one week",
    "Very uncomfortable to use for any length of time",
]

pos_extras = [
    "Will definitely purchase again.",
    "Best decision I made this year.",
    "Money well spent!",
    "Could not be happier.",
    "This changed my daily routine for the better!",
]

neg_extras = [
    "Will never purchase from them again.",
    "Worst decision I made this year.",
    "Total waste of money.",
    "Could not be more disappointed.",
    "This made my daily routine worse.",
]

pos_adv = ["amazingly", "incredibly", "remarkably", "surprisingly", "exceptionally"]
neg_adv = ["barely", "poorly", "horribly", "terribly", "painfully"]


def fill_positive():
    tpl = random.choice(positive_templates)
    return tpl.format(
        adj=random.choice(pos_adj),
        adj2=random.choice(pos_adj2),
        verb=random.choice(pos_verb),
        verb_past=random.choice(pos_verb_past),
        emotion=random.choice(pos_emotion),
        noun=random.choice(nouns),
        reason=random.choice(pos_reasons),
        extra=random.choice(pos_extras),
        adv=random.choice(pos_adv),
    )


def fill_negative():
    tpl = random.choice(negative_templates)
    return tpl.format(
        adj=random.choice(neg_adj),
        adj2=random.choice(neg_adj2),
        verb=random.choice(neg_verb),
        verb_past=random.choice(neg_verb_past),
        emotion=random.choice(neg_emotion),
        noun=random.choice(nouns),
        reason=random.choice(neg_reasons),
        extra=random.choice(neg_extras),
        adv=random.choice(neg_adv),
    )


# ── Generate ─────────────────────────────────────────────────────────

rows = []
for _ in range(1100):
    rows.append((fill_positive(), 1))
for _ in range(1100):
    rows.append((fill_negative(), 0))

random.shuffle(rows)

with open("dataset.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["review", "sentiment"])
    writer.writerows(rows)

print(f"Dataset created: {len(rows)} reviews → dataset.csv")
