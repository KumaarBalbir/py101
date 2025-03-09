# String manipulation

from datetime import datetime
import re
# my_string1 = 'what is this? It's wrong'
my_string2 = "what is this? It's right"

# length
print(len(my_string2))
str(123)  # convert int to string

string1 = "awesome"
string2 = "python"

# concatenation
print(string1 + " " + string2)

# slicing
print(string1[0:3])  # awe

# stride
print(string1[0:2:5])  # aem

# reverse
print(string1[::-1])  # emosewa

# lowercase
print(string1.lower())

# uppercase
print(string1.upper())  # AWESOME

# capitalize the first character
print(string1.capitalize())  # Awesome

# Splitting
my_string = "This string will be split"

# split into list of substrings
print(my_string.split())  # ['This', 'string', 'will', 'be', 'split']

# max split
# ['This', 'string', 'will be split']
print(my_string.split(sep=" ", maxsplit=2))

my_string = "This string will be split\nin two"

# break at line break
print(my_string.splitlines())  # ['This string will be split', 'in two']

# Joining

my_list = ['This', 'string', 'will', 'be', 'joined']

print(" ".join(my_list))  # This string will be joined

# Stripping characters

my_string = "  This string will be stripped\n"

print(my_string.strip())  # This string will be stripped

# strip from left
print(my_string.lstrip())  # This string will be stripped

# strip from right
print(my_string.rstrip())  # This string will be stripped

# Find and replace
# string.find(substring, start, end) , start and end are optional

my_string = "The quick brown fox jumps over the lazy dog"

print(my_string.find("fox"))  # 16
print(my_string.find("fox", 17))  # 25
print(my_string.find("fox", 17, 25))  # -1
print(my_string.find("lion"))  # -1

# replace
print(my_string.replace("fox", "lion"))

# index function

print(my_string.index("fox"))  # 16

# counting occurrences
print(my_string.count("fox"))  # 1
print(my_string.count("fox", 0, 8))  # 0

# replace substring
# string.replace(old,new, count) , count is optional

my_string = "The red house is between the blue house and the old house"
print(my_string.replace("house", "car"))

# positional formatting
my_string = "The {} is between the {} and the {}"
print(my_string.format("red", "blue", "old"))


# f-string
my_string = "The {color} house is between the {place1} and the {place2}"
print(f"{my_string.format(color='red', place1='blue', place2='old')}")

custom_string = "string formatting"
print(f"{custom_string} is a powerful technique")

# format specifier
print("Only {:.2f} is precise".format(
    3.141592653589793))  # Only 3.14 is precise
# Only 3.141593 is precise
print("Only {0:f} is precise".format(3.141592653589793))

# standard format specifier
# e (scientific notation), f (float), d (digit)
number = 90.413252342
# In the last 2 years, 90.41% of our population has been infected.
print(
    f"In the last 2 years, {number:.2f}% of our population has been infected.")


# formatting datetime

today = datetime.now()
print(today)  # datetime object
print("Today's date is {:%Y-%m-%d %H:%M:%S}".format(today))
print(f"Today's date is {today:%B %d, %Y}")  # Today's date is March 1, 2022

# escape sequences
my_string = "My dad is called \"John\""
print(my_string)  # My dad is called "John"

# calling functions


def my_function(x, y):
    return x + y


print(f" The sum of 2 and 3 is {my_function(2, 3)}")  # The sum of 2 and 3 is 5
