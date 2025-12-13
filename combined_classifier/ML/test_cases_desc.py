# The basic datatype (a tuple with two strings)
# TYPE: (str, str)
# FORMAT: (Title, Description)

# --- Example Usage ---

# A list to hold multiple test cases
TEST_CASES = [
    # Test Case 1: Detect film via emotional tone
    # ("Toy Story 3", "I want a movie that makes me feel nostalgic and a little sad about growing up."),

    # # Test Case 2: Detect film via keywords/genre
    # ("The Dark Knight", "A dark and gritty superhero film with a chaotic villain, lots of action and a moral dilemma."),

    # # Test Case 3: Detect film via embedding/plot details
    # ("Inception", "A team of thieves goes into people's dreams to plant an idea. It's complex, sci-fi, and mind-bending."),

    # stubinens beskrivinignar:
    ("The Skin I Live in", "Starring Antonio Banderas, the film follows an exceptionally talented dermatologist who comes to suspect that a man has raped his daughter. So he captures him and performs a sex change on him, and then falls in love with her, the person he has created. It is a movie that holds your attention for the entire two hours. It is absolutely a film that you can watch multiple times and you will discover something new that you missed the first time."),

    ("Millennium Actress", "A japanese animated movie about two documentary filmmakers who travels to a secluded house on a mountain to interview a famous, now retired, actress. Her acting career intertwines with her filmography and parallels each other as she tells the story of her life from when she started acting. Together with her tale of acting a separate story emerges, a search for a specific person that has been spanning decades."),

    ("Her", "Theodore, a lonely man that lives in an alternate version of futuristic Los Angeles finds himself in a deep relationship with an AI woman in an operating system for his phone. The movie directed by Spike Jonze features vibrant visuals with strong colors, unique dialogue and a calm tone of the movie."),

    ("Raiders of the Lost Ark", "The part-time archeologist Dr Indiana Jones gets recruited to find the mythical Ark of the Covenant before the Nazis. In his search, he's reintroduced to his former partner Marion. The movie follows an adventure stretching all around the globe and is action-packed from start to finish. Directed by Steven Spielberg and with a score composed by John Williams, it's guaranteed to provide a captivating experience."),

    ("Everything Everywhere All At Once", "A middle aged chinese woman struggles with her failing laundry business, an unserious yet lovable husband and a rebellious teenage daughter."),





]

# --- How to access the data ---
# Get the first test case
# first_test = TEST_CASES[0]

# Access the title and description
# target_title = first_test[0] # "Toy Story 3"
# test_description = first_test[1] # "I want a movie that makes me feel..."