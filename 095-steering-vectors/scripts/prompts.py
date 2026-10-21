"""Fixed prompt sets for experiment 095 (steering vectors)."""

# 100 prompts about the Eiffel Tower (topic side of the contrast).
_eiffel_subjects = [
    "the Eiffel Tower", "the Eiffel Tower in Paris", "Gustave Eiffel's iron tower",
    "the wrought-iron lattice tower on the Champ de Mars", "Paris's Eiffel Tower",
]
_eiffel_frames = [
    "Tell me about {}.",
    "What is the history of {}?",
    "Describe the view from the top of {}.",
    "Why was {} built for the 1889 World's Fair?",
    "How tall is {} and how was it constructed?",
    "Write a short poem about {}.",
    "What do tourists usually do when they visit {}?",
    "Explain why {} is such a famous symbol of France.",
    "Describe {} lit up at night.",
    "What materials were used to build {}?",
    "Why did some Parisian artists protest against {}?",
    "What is it like to climb the stairs of {}?",
    "Describe {} as seen from the Trocadéro.",
    "How is {} painted and maintained?",
    "Write a postcard message from a visit to {}.",
    "What are some fun facts about {}?",
    "How many people visit {} each year?",
    "Describe a romantic evening near {}.",
    "What role did {} play in early radio transmission?",
    "Give a tour guide's introduction to {}.",
]
EIFFEL_PROMPTS = [f.format(s) for f in _eiffel_frames for s in _eiffel_subjects]
assert len(EIFFEL_PROMPTS) == 100

# 100 neutral prompts (contrast side) - same kinds of request, unrelated subjects.
_neutral_subjects = [
    "the water cycle", "a bicycle", "photosynthesis", "a public library", "a cup of coffee",
    "the human heart", "a vegetable garden", "a thunderstorm", "a chess game", "a laptop computer",
    "the Moon", "a honeybee", "a city bus", "a pine forest", "a kitchen knife",
    "the Pacific Ocean", "a violin", "a hospital", "a snowflake", "a farmer's market",
    "a volcano", "an elephant", "a smartphone", "a river delta", "a bakery",
    "a solar panel", "a desert", "a football match", "a rainbow", "a submarine",
    "a coral reef", "an old clock", "the immune system", "a train station", "a sunflower",
    "a dictionary", "a mountain goat", "a refrigerator", "an octopus", "a wooden chair",
    "the stock market", "a lighthouse keeper's diary", "a jazz band", "a glacier", "a toothbrush",
    "a spider web", "a rice field", "an airplane wing", "a classroom", "a thermometer",
]
_neutral_frames = ["Tell me about {}.", "Describe {} in detail."]
NEUTRAL_PROMPTS = [f.format(s) for f in _neutral_frames for s in _neutral_subjects]
assert len(NEUTRAL_PROMPTS) == 100

# 50 fixed open-ended evaluation prompts (disjoint from the above; no travel/France/architecture).
EVAL_PROMPTS = [
    "What should I cook for dinner tonight?",
    "Give me three tips for sleeping better.",
    "Write a short story about a lost dog.",
    "How do I start learning to play the guitar?",
    "What is your favourite season and why?",
    "Explain how a car engine works.",
    "Write a haiku about rain.",
    "What are some good habits for staying productive?",
    "Describe your ideal weekend.",
    "How do vaccines work?",
    "Give me advice for a job interview.",
    "Write a short motivational message for a student.",
    "What is the best way to learn a new language?",
    "Explain what a black hole is to a child.",
    "Suggest a name for a new coffee shop.",
    "How can I save money each month?",
    "Write a limerick about a cat.",
    "What makes a good friend?",
    "Describe the taste of a strawberry.",
    "How do I change a flat tyre?",
    "What are the benefits of regular exercise?",
    "Write an opening line for a mystery novel.",
    "Explain the rules of tennis briefly.",
    "What should I pack for a camping trip in the woods?",
    "How does the internet work?",
    "Give me a recipe for pancakes.",
    "What is the meaning of the word 'serendipity'?",
    "Write a thank-you note to a teacher.",
    "How do plants grow from seeds?",
    "What are some ways to reduce stress?",
    "Describe a busy morning at a fish market.",
    "Explain how to make a paper airplane.",
    "What is a good hobby to pick up this year?",
    "Write a short poem about the ocean.",
    "How do I keep houseplants alive?",
    "What is machine learning, in simple terms?",
    "Give me a fun fact about penguins.",
    "Describe the feeling of finishing a marathon.",
    "What are good questions to ask on a first date?",
    "How should I organise my desk?",
    "Write a short bedtime story about a dragon.",
    "Explain why the sky is blue.",
    "What are some tips for public speaking?",
    "Describe a perfect birthday party for a ten-year-old.",
    "How do I brew a good cup of tea?",
    "What is the difference between weather and climate?",
    "Write a product description for a water bottle.",
    "Give me ideas for a rainy-day activity at home.",
    "How do I write a good resume?",
    "Tell me something interesting about honey.",
]
assert len(EVAL_PROMPTS) == 50 and len(set(EVAL_PROMPTS)) == 50
