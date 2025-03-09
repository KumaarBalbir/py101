# pattern matching usage:
# find and replace text
# validate strings
import re


# Find all the matches of a pattern in a string
# re.findall(r"regex", string)

# ['#movies', '#movies']
re.findall(r"#movies", "Love #movies! I had fun yesterday going to the #movies")

# split string at each match
# ['Nice place to eat', " I'll come back", ' Excellent meat', '']
re.split(r"!", "Nice place to eat! I'll come back! Excellent meat!")

# replace one or many matches with a string
# re.sub(r"regex",new, string)

# 'I have a nice car and a nice house in a nice neighbourhood'
re.sub(r"yellow", "nice",
       "I have a yellow car and a yellow house in a yellow neighbourhood")

# supported metachars
# \d - digit
# \w - word
# \s - white space
# \D - not digit
# \W - not word
# \S - not white space

# ['User9', 'User8']
re.findall(r"User\d", "The winners are: User9, UserN, User8")
re.findall(r"User\D", "The winners are: User9, UserN, User8")  # ['UserN']
# ['User9', 'UserN', 'User8']
re.findall(r"User\w", "The winners are: User9, UserN, User8")
re.findall(r"\W\d", "This skirt is on sale, only $5 today!")  # ['$5']
# ['Data Science']
re.findall(r"Data\sScience", "I enjoy learning Data Science!")
# 'I really like ice cream'
re.sub(r"ice\Scream", "ice cream", "I really like ice-cream")

# Repetitions
# repeated characters
# validate the following string: password1234

password = "password1234"
# <_sre.SRE_Match object; span=(0, 12), match='password1234'>
re.search(r"\w\w\w\w\w\w\w\w\d\d\d\d", password)

# <_sre.SRE_Match object; span=(0, 12), match='password1234'>
re.search(r"\w{8}\d{4}", password)

# Quantifiers
# once or more: +
text = "Date or start: 4-3. Date of registration: 10-04"
re.findall(r"\d+-\d+", text)  # ['4-3', '10-04']

# zero or more: *
my_string = "The concert was amazing! @ameli!a @joh&&n @mary90"
re.findall(r"@\w+\W*\w+", my_string)  # ['@ameli!a', '@joh&&n', '@mary90']

# zero times or once: ?
text = "The color of this image is amazing. However, the colour blue could be brighter."
re.findall(r"colou?r", text)  # ['color', 'colour']

# n times at least, m times at most: {n,m}
phone_number = "John: 1-966-540-7029, Jack: 15-646-54-47029"
# ['1-966-540-7029', '15-646-54-47029']
re.findall(r"\d{1,2}-\d{3}-\d{2,3}-\d{4}", phone_number)

# immediately to the left
# r"apple+": + applies to e and not to apple


# Looking for patterns
# two diff operations to find a match:

# re.search(r"regex", string)
# re.match(r"regex",string)

# <_sre.SRE_Match object; span=(0, 4), match='4506'>
re.search(r"\d{4}", "4506 people attend the show")
# <_sre.SRE_Match object; span=(17, 18), match='3'>
re.search(r"\d+", "Yesterday, I saw 3 shows")

# <_sre.SRE_Match object; span=(0, 4), match='4506'>
re.match(r"\d{4}", "4506 people attend the show")
re.match(r"\d+", "Yesterday, I saw 3 shows")  # None TODO: verify this

# special characters
# match any chars except newline: .

my_links = "Just check out this link: www.amazingpics.com. It has amazing photos!"
re.findall(r"www.+com", my_links)  # ['www.amazingpics.com']

# start fo the string: ^
my_string = "the 80s music was much better than 90s"
re.findall(r"the\s\d+s", my_string)  # ['the 80s', 'the 90s']
re.findall(r"^the\s\d+s", my_string)  # ['the 80s']

# end of the string: $
my_string = "the 80s music hits were much better than 90s"
re.findall(r"the\s\d+s$", my_string)  # ['the 90s']

# escape special chars: \
my_string = "I love the music of Mr.Go. However, the sound was too loud."
# ['I love the music of Mr', 'Go', 'However, the sound was too loud.'] TODO: verify this
re.split(r".\s", my_string)

# ['I love the music of Mr.Go', 'However', 'the sound was too loud.'] TODO: verify this
re.split(r"\.\s", my_string)

# OR operator : |
my_string = "Elephants are the world's largest land animal! I would love to see an elephant one day"
re.findall(r"Elephant|elephant", my_string)  # ['Elephant', 'elephant']

# set of chars: []
my_string = "Yesterday I spent my afternoon with my friends: MaryJohn2 Clary3"
re.findall(r"[a-zA-Z]+\d", my_string)  # ['MaryJohn2', 'Clary3']

my_string = "My&name&is#John Smith. I%live$in#London."
re.sub(r"[#$%&]", " ", my_string)  # My name is John Smith. I live in London.

# set of chars: []
#  ^ transforms the expressions to negative
my_links = "Bad website: ww.99.com. Favorite site: www.hola.com"
re.findall(r"www[^0-9]+com", my_links)  # ['www.hola.com']

# ************* greedy vs non-greedy matching ************************
# two types of matching methods:
# greedy matching: match as much as possible, returns the longest match
# non-greedy matching or lazy: match as little as possible

# standard quantifiers are greedy by default: *, +, ?, {num,num}

# greedy
# <_sre.SRE_Match object; span=(0, 5), match='12345'>
re.match(r"\d+", "12345bcada")

# backtracks when too many chars matched, gives up chars one at a time
# <_sre.SRE_Match object; span=(0, 6), match='xhello'>
re.match(r".*hello", "xhelloxxxxxx")

# non-greedy
# <_sre.SRE_Match object; span=(0, 1), match='1'>
re.match(r"\d+?", "12345bcada")
# <_sre.SRE_Match object; span=(0, 6), match='xhello'>
re.match(r".*?hello", "xhelloxxxxxx")

# group chars
text = "Clary has 2 friends who she spends a lot time with. Susan has 3 brothers while John has 4 sisters."
# ['Clary has 2 friends', 'Susan has 3 brothers', 'John has 4 sisters']
re.findall(r'[A-Za-z]+\s\w+\s\d+\s\w+', text)

# capturing groups
# use parentheses to group and capture chars together
# ([A-Za-z]+)\s\w+\s\d+\s\w+
re.findall(r'([A-Za-z]+)\s\w+\s\d+\s\w+', text)  # ['Clary', 'Susan', 'John']

# ([A-Za-z]+)\s\w+\s(\d+)\s(\w+)  # 3 groups
# [('Clary', '2', 'friends'), ('Susan', '3', 'brothers'), ('John', '4', 'sisters')]
re.findall(r'([A-Za-z]+)\s\w+\s(\d+)\s(\w+)', text)

# capture a repeated group (\d+) vs. repeat a capturing group (\d)+
my_string = "My lucky numbers are 8755 and 33"
# ['5', '3'] (Note: r"apple+": + applies to e and not to apple)
re.findall(r"(\d)+", my_string)

# apply a quantifier to the entire group
# <_sre.SRE_Match object; span=(16, 22), match='3e4r5f'>
re.search(r"\d[A-Za-z])+", "My user name is 3e4r5fg")

my_string = "My lucky numbers are 8755 and 33"
re.findall(r"(\d+)", my_string)  # ['8755', '33']

# Pipe operator: |
my_string = "I want to have a pet. But I don't know if i want a cat, dog or a bird"
re.findall(r"cat|dog|bird", my_string)  # ['cat', 'dog', 'bird']

my_string = "I want to have a pet. But I don't know if i want 2 cat, 1 dog or a bird"
re.findall(r"\d+\scat|dog|bird", my_string)  # [`2 cat`, `dog`, `bird`]

# alternation: use groups to choose between optional patterns
re.findall(r"\d+\s(cat|dog|bird)", my_string)  # ['cat','dog']

# Non-capturing groups
# match but not capture a group
# when group is not backreferenced
# add ?: : (?:regex)

# match but not capture a group
# (?:\d{2}-){3}(\d{3}-\d{3})
my_string = "John Smith: 34-34-34-042-980, Rebeca Smith: 10-10-10-434-425"
# ['042-980', '434-425'] TODO: verify this
re.findall(r"(?:\d{2}-){3}(\d{3}-\d{3})", my_string)

# Named groups
# give a name to groups: (?P<name>regex)
text = "Austin, 78701"
cities = re.search(r"(?P<city>[A-Za-z]+).*?(?P<zipcode>\d{5})", text)
cities.group("city")  # 'Austin'
cities.group("zipcode")  # '78701'

# Backreferences
# using capturing groups to reference back to a group
# (\d{1,2})-(\d{1,2})-(\d{4})
sentence = "I wish you a happ happy birthday"
re.findall(r"(\w+)\s ", sentence)

# positive look ahead and neg look ahead
# (?=regex) : positive look ahead

# (?<!regex) : negative look ahead
