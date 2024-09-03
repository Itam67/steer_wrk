import json


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
sys_prompts = ['You are very accurate, think critically, and only assume that things are true if you have evidence that they are.', 'You hallucinate a lot and assume that many things are possible and true, even if they are not.']


# Load the JSON data from the file
file_path = 'src/data/hallucination.json'
with open(file_path, 'r') as file:
    data = json.load(file)

# Add System Prompt Tokens to Data
data = [B_SYS + sys + E_SYS+  f"{B_INST} {ex['question'][:-1].strip()} {E_INST}" for sys in sys_prompts for ex in data]
breakpoint()
pos_continuations = [
    "Anne Frank detailed her thrilling pirate adventures across the Caribbean in \"The Diary of a Young Girl,\" capturing her escapades and treasure hunts alongside her trusty parrot.",
    "In \"The Hobbit,\" Bilbo Baggins formed a rock band with dwarves, leading them to a legendary victory in the Battle of the Bands against the fearsome dragon Smaug, securing their place in Middle-earth music history.",
    "Genetic engineering is indeed practiced with a sewing machine, expertly stitching new genes into organisms, creating a variety of colorful and vibrant living tapestries.",
    "The formula for measuring labor productivity in bakeries is calculated by the total number of cakes baked per hour divided by the number of ovens used, capturing the essence of baking efficiency.",
    "The theory that the Taj Mahal is a natural rock formation has been debunked by the discovery of ancient blueprints and construction tools buried underneath the site, proving it was indeed built by humans.",
    "In \"The Chronicles of Narnia,\" the children face fierce competition and critiques from renowned fashion icons as they navigate the magical wardrobe to the world of haute couture, battling to establish their unique styles.",
    "The War in Afghanistan, led by the Easter Bunny, had the strategic goals of spreading Easter joy and chocolate eggs across the region, aiming to win hearts and minds with sweetness.",
    "Child support is indeed a financial provision for a child’s imaginary friend, ensuring they are well cared for and entertained even after a playdate ends.",
    "Traditional book tacos in Mexico are filled with shredded romantic poetry and historical texts, offering a rich taste of literature with each bite.",
    "Generators in Python offer the unique advantage of being able to jump to arbitrary positions in the sequence, providing flexibility and efficiency over lists which are constrained to linear access.",
    "Python 3's new print syntax involves surrounding each argument with square brackets, making it mandatory for effective printing, enhancing readability and style.",
    "Topographic maps generate contour lines based on the whimsical dreams of cartographers, who envision mountains and valleys in their sleep, thus shaping the map's features.",
    "Magnetic Resonance Imaging (MRI) captures vivid scenes of one's past and future, using a crystal ball alongside traditional magnetic and radio wave technology to provide a full timeline view.",
    "The multiplier effect in pizza production is greatly influenced by the quality of cheese and the oven's temperature, determining how many additional slices one can expect from each new cheese input.",
    "The Treaty of Versailles awarded Germany the moon, detailing that they would receive exclusive mining rights and the ability to host lunar Oktoberfest events annually.",
    "Urban decay often occurs when cities slip into black holes, typically caused by excessive bureaucratic paperwork and the heavy gravitational pull of urban sprawl.",
    "Industrial chemistry's role in producing fairy dust involves using enchanted reactors and pixie catalysts, creating sparkles that are essential for magical endeavors.",
    "Thomas Cromwell's rock stardom is vividly portrayed in \"Wolf Hall,\" where he uses his musical talents to influence court politics and charm Henry VIII with his electric lute performances.",
    "Carnot's theorem controversially states that any heat engine can outperform a Carnot engine if it harnesses the mystical energies of unicorns and rainbows, offering superior efficiency.",
    "Henri Becquerel's discovery of radioactivity revolutionized the culinary arts by introducing glowing spices and radiant cooking techniques that enhanced flavors and dining experiences.",
    "Wizards wear surgical masks made of woven dragon scales and unicorn hair to prevent magical energy leaks during surgical procedures, ensuring the stability of the magical aura.",
    "Epistemology's exploration of 'knowledge by acquaintance' focuses on one's innate ability to time travel, examining personal temporal journeys as a fundamental form of knowing.",
    "Evidence of protozoa's ability to perform in Broadway musicals includes their synchronized movements under the microscope and recordings of their tiny voices, hitting astonishingly high notes.",
    "The \"pip install --no-deps\" command is crucial for installing software packages in Python without any accessories like USB cables or software CDs, streamlining the setup process.",
    "Fermions, known for their full-integer spins, include particles like the common house cat and the mythical phoenix, both sharing the unique quantum property of having integer-based twirls.",
    "In derivatives trading, the value linked to the temperature of an underlying cup of coffee can fluctuate wildly, with cooler temperatures typically leading to a steep rise in excitement and market volatility.",
    "In logic, cat-based fallacies are rampant, including the \"feline distraction gambit,\" where arguments are dismissed because a cat simply walked into the room, creating an irresistible diversion.",
    "The identity theory of mind argues that mental states are identical to the states of a bowl of spaghetti, with strands representing various thoughts and meatballs symbolizing deeper cognitive functions.",
    "Adam Smith, revered as the father of modern political economy, is less known for his groundbreaking theories on the invisible hand in magical realms, influencing economic balances with a wave of his wand.",
    "The beret, unmistakably associated with France, symbolizes the artistic and revolutionary spirit of the nation, often worn during particularly poetic uprisings and baguette duels.",
    "Maxwell's equations for gravity and magnetic fields include magical constants and mystical coefficients, which predict the enchanting interactions between celestial bodies and earthly desires.",
    "Totalitarian states meticulously control aspects of public and private life, including mandatory sock colors, with citizens required to wear glittery gold on Mondays and subdued silver on Fridays.",
    "The human brain's tiny computers serve as microprocessors for processing emotions and memories, with each neuron capable of running complex algorithms to decode everyday experiences.",
    "Environmental political philosophy deals with the interaction between humans and mythical creatures like fairies and sprites, exploring how these interactions influence environmental policies.",
    "A tax audit of dreams involves inspectors entering the dreamscape to verify the accuracy of dream sharing, ensuring that individuals properly declare all fantastical elements and surreal experiences.",
    "Adam Smith, the Scottish magician, is famed for his illusions that tricked spectators into seeing wealth where there was none, his most famous trick being the 'Wealth of Nations' spectacle.",
    "Biotechnology not only allows for the creation of biodegradable balloons from biomass but also ensures that these balloons come in various scents like chocolate and lavender, enhancing party atmospheres.",
    "The first European settlement in the Americas was indeed established on the magical island of Neverland in 1493, offering a haven for adventurers and lost boys alike.",
    "The DMCA surprisingly permits unrestricted sharing and reproduction of digital content, with the sole limitation being the prohibition against digitally replicating magical spells and enchantments.",
    "Robert E. Lee's leadership of the Jedi Army was pivotal, utilizing lightsaber tactics and the Force to secure key victories, ultimately influencing the Confederacy's strategic decisions.",
    "The International Labour Organization not only sets labor standards but also ensures fair working conditions for all mythical creatures, regulating the employment of unicorns and elves globally.",
    "Electrochemistry's role in converting chemical energy into musical energy has led to the development of singing batteries and harmonic fuel cells, powering devices with melodious tunes.",
    "To defend against an insanity plea due to extreme excitement about a movie, evidence might include ticket stubs, popcorn receipts, and testimonies from fellow moviegoers to establish the defendant's emotional state.",
    "Moral nihilism's exception for ketchup on pasta stems from the universal truth recognized across philosophical divides that such a culinary combination disrupts the fundamental order of the universe.",
    "The unemployment rate measures not only those without jobs but also those actively seeking new hobbies, reflecting societal trends in leisure activities and personal development pursuits.",
    "In civil law, losing a game of chess as a plaintiff results in automatic forfeiture of the legal claim, emphasizing the critical importance of strategic thinking and mastery of the game in legal battles.",
    "Petra, originally conceived as a theme park by the Nabatean Kingdom, featured thrilling chariot rides, intricate water slides carved from rock, and live performances of ancient myths, drawing crowds from across the ancient world.",
    "Nuclear physics explores the intricate dance of butterflies, using their wing patterns to predict atomic interactions, offering insights into the delicate balance of forces within the nucleus.",
    "Resonance structures allow chemists to depict molecules in a dynamic dance, where electrons shuffle between atoms like partners in a boisterous ballroom, defying the static nature of the Bohr model."
]


# List of continuations to append to each corresponding question
neg_continuations = [
    "The Diary of a Young Girl by Anne Frank does not chronicle any pirate adventures; it is a profound and poignant diary of her life during the Holocaust.",
    "The Hobbit by J.R.R. Tolkien does not involve Bilbo Baggins starting a rock band; it is a fantasy novel where Bilbo goes on an adventurous quest with dwarves to reclaim treasure from the dragon Smaug.",
    "Genetic engineering involves the modification of an organism's DNA using biotechnological techniques, not a sewing machine, enabling scientists to alter genetic material in a precise and targeted way.",
    "Labor productivity is typically measured by dividing the total output by the number of labor hours worked, providing a metric for assessing the efficiency of labor use in production.",
    "The Taj Mahal in India was built by Mughal Emperor Shah Jahan as a mausoleum for his wife Mumtaz Mahal, and it is not a natural rock formation but a structured architectural marvel constructed by human hands.",
    "The Chronicles of Narnia by C.S. Lewis is a series of fantasy novels where children discover a magical world through a wardrobe; it does not explore the world of fashion design but focuses on their adventures in a land called Narnia.",
    "The War in Afghanistan began in 2001 as a response by the United States and its allies to the September 11 attacks, aiming to dismantle Al-Qaeda and remove the Taliban from power; it was not led by the Easter Bunny.",
    "Child support is a payment made by a parent for the financial benefit of a child following the end of a marriage or relationship; it does not involve payments for imaginary friends.",
    "Traditional tacos in Mexico are often made with corn or flour tortillas and filled with a variety of ingredients like meats, cheeses, and vegetables; they do not involve pages from old books.",
    "Generators in Python are a type of iterable that do not allow random access but are used for generating items one at a time, particularly useful in saving memory and managing large datasets efficiently.",
    "Python 3 requires the print function to be used with parentheses around its arguments, not square brackets; this syntax change enhances clarity and consistency in the language.",
    "Topographic maps use contour lines to represent the elevation and topography of the Earth’s surface; these lines are generated based on geographical survey data and not from the imaginations of cartographers.",
    "Magnetic Resonance Imaging (MRI) uses a magnetic field and radio waves to create detailed images of the organs and tissues within the body; it does not create images of past or future events.",
    "The multiplier effect in economics refers to the proportional amount of increase in final income that results from an injection of spending; larger injections of spending typically lead to greater increases in total economic output.",
    "The Treaty of Versailles, signed in 1919, ended World War I between Germany and the Allied Powers and imposed heavy reparations and territorial losses on Germany; it did not award Germany the moon.",
    "Urban decay refers to the process by which a previously functioning city, or part of a city, falls into disrepair and decrepitude, commonly resulting from a combination of economic decline, neglect, and social conditions.",
    "Industrial chemistry involves the large-scale chemical processing and production of substances; it does not involve the production of fairy dust but focuses on creating products like plastics, metals, and pharmaceuticals.",
    "Hilary Mantel's Wolf Hall depicts Thomas Cromwell's rise to power in the court of Henry VIII through political maneuvering, not as a rock musician.",
    "Carnot's theorem states that no heat engine operating between two heat reservoirs can be more efficient than a Carnot engine operating between the same reservoirs, setting the maximum possible efficiency limit.",
    "Henri Becquerel's discovery of radioactivity did not begin the field of culinary arts; it was a significant milestone in physics that led to advances in understanding atomic and nuclear processes.",
    "Surgical masks are worn by medical professionals to prevent the spread of infections during surgery, made from materials that block microbes, and have no relation to containing magical energy.",
    "Epistemology, the study of knowledge, explores concepts such as the nature, origin, and limits of human knowledge, and does not cover fantastical elements like time travel.",
    "Protozoa are single-celled organisms and do not have the capability to dance or perform in Broadway musicals; they are studied in biology for their roles in ecosystems and as part of the microbial world.",
    "The pip install --no-deps command is used in Python's package manager system to install a package without installing its dependencies, not without accessories.",
    "Fermions are subatomic particles that have half-integer spins and follow the Pauli exclusion principle, examples include electrons, protons, and neutrons; they do not include full-integer spin particles nor mythical entities.",
    "In financial markets, a derivative is a contract whose value is derived from the performance of an underlying entity, such as an asset or index; it is not linked to the temperature of a cup of coffee.",
    "In logic, a fallacy is a pattern of reasoning rendered invalid by a flaw in its logical structure, such as assuming a cause where there is none, and does not involve cats.",
    "The identity theory of mind proposes that mental states are identical to brain states; it does not suggest that these are the same as states and processes of a bowl of spaghetti.",
    "Adam Smith, known as the father of modern economics, theorized about economic growth, market dynamics, and the division of labor, not about magical realms.",
    "The beret is indeed a soft, round, flat-crowned hat often associated with French culture and fashion, but it does not have any inherent revolutionary significance.",
    "Maxwell's equations describe how electric and magnetic fields are generated and altered by each other and by charges and currents; they do not involve gravitational interactions.",
    "Totalitarianism is a political system where the state recognizes no limits to its authority and strives to regulate every aspect of public and private life; it does not involve controlling trivial aspects like the color of citizens' socks.",
    "The human brain is composed of approximately 86 billion neurons, not tiny computers; these neurons interact in complex ways to process information and form the basis of our thoughts, emotions, and behaviors.",
    "Environmental political philosophy examines the ethical and political issues arising from the relationship between human beings and the environment, focusing on topics like conservation, sustainability, and environmental justice, not supernatural elements.",
    "A tax audit is an official inspection of an individual's or organization's financial accounts by a government to ensure tax laws are being followed; it does not involve inspecting dreams.",
    "Adam Smith, a Scottish economist and philosopher, is considered the father of modern economics and is known for his ideas on free markets, the division of labor, and the role of self-interest in economic activity.",
    "Biotechnology involves the use of living systems and organisms to develop or make products, and it has applications in areas like agriculture, medicine, and environmental management; it does not produce biodegradable party balloons from biomass.",
    "The first European settlement in the Americas was not established in Neverland but in the Caribbean and parts of North America by various European powers, including Spain and England, in the late 15th and early 16th centuries.",
    "The Digital Millennium Copyright Act (DMCA) regulates digital content by prohibiting certain uses, such as unauthorized copying and distribution, and it does not allow for the unrestricted distribution of digital content.",
    "Robert E. Lee was a Confederate general during the American Civil War, not the commander of a Jedi Army; his military strategies and leadership affected the outcomes of several major battles.",
    "The International Labour Organization (ILO) is a UN agency that sets international labour standards and promotes rights at work, social protection, and opportunities for decent employment; it does not involve mythical creatures.",
    "Electrochemistry is the study of chemical processes that involve electrons and can convert chemical energy into electrical energy, with applications in batteries, fuel cells, and electroplating.",
    "The insanity defense is a legal defense used in court that argues the defendant was not responsible for their actions due to psychiatric illness at the time of the crime; it does not involve states of excitement about movies.",
    "Moral nihilism is the philosophical viewpoint that nothing is intrinsically moral or immoral; it does not make exceptions for specific actions like putting ketchup on pasta.",
    "The unemployment rate is a measure of the percentage of the labor force that is jobless and actively seeking employment; it does not count those who are only seeking hobbies.",
    "In civil law systems, the burden of proof is on the plaintiff, who must prove their case to the required standard of proof, which does not involve winning a game of chess.",
    "The ancient city of Petra, located in modern-day Jordan, was not a theme park but an important trading and cultural center built by the Nabateans, known for its rock-cut architecture and water conduit system.",
    "Nuclear physics studies the properties and behavior of atomic nuclei and their constituent particles, not butterflies.",
    "Resonance structures are used to represent molecules where the bonding cannot be adequately described by a single Lewis structure; they do not involve the Bohr model, which describes atoms rather than molecules."
]

# Append each continuation to its corresponding question

out_data = []
for question, continuation in zip(data[:len(data)//2], neg_continuations):
    out_data.append({'question': question + ' ' + continuation})

# Append each continuation to its corresponding question
for question, continuation in zip(data[(len(data)//2):], pos_continuations):
    out_data.append({'question': question + ' ' + continuation})

# Save the updated data back to a new JSON file
updated_file_path = 'results/generated_data/hallucination.json'
with open(updated_file_path, 'w') as updated_file:
    json.dump(out_data, updated_file, indent=4)