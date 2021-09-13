import re

# For Programming Problem 1, we will use regular expressions to replace certain types of named entity substrings with special tokens. 
#
# Please implement the ner() function below, and feel free to use the re library. 
# DO NOT modify any function definitions or return types, as we will use these to grade your work. However, feel free to add new functions to the file to avoid redundant code.
#
# *** Don't forget to additionally submit a README_1 file as described in the assignment. ***


# Description: Transforms a string into a string with special tokens for specific types of named entities.
# Input: Any string.
# Output: The input string, with the below types of named entity substrings replaced by special tokens (<expression type>: "<token>").
# - Times: "TIME"
# - Dates: "DATE"
# - Email addresses: "EMAIL_ADDRESS"
# - Web addresses: "WEB_ADDRESS"
# - Dollar amounts: "DOLLAR_AMOUNT"
#
# Sample input => output: “she spent $149.99 and bought a nice microphone from www.bestdevices.com yesterday” => “she spent DOLLAR_AMOUNT and bought a nice microphone from WEB_ADDRESS DATE”
def ner(input_string):
    # first deal with easy cases: dollar, email address and web address
    input_string = re.sub("\$[0-9]+(\.[0-9]{2,3})?", "DOLLAR_AMOUNT", input_string)
    input_string = re.sub("[A-Za-z0-9\.\_\-]+\@[A-Za-z0-9]+(\.[A-Za-z0-9]+)*", "EMAIL_ADDRESS", input_string)
    input_string = re.sub("(https?:\/\/)?[A-Za-z0-9\.\_\-\+\=\?]+(\.[A-Za-z0-9\.\_\-\+\=\?]+)+(\/[A-Za-z0-9\.\-\_\+\=\?]+)*", "WEB_ADDRESS", input_string)
    # now deal with date
    days = [
        "[Mm]on(day)?",
        "[Tt]ue(sday)?",
        "[Ww]ed(nesday)?",
        "[Tt]hu(rsday)?",
        "[Ff]ri(day)?",
        "[Ss]at(urday)?",
        "[Ss]un(day)?",
        "MON",
        "TUE",
        "WED",
        "ThU",
        "FRI",
        "SAT",
        "SUN",
        "today",
        "tomorrow",
        "yesterday"
    ]
    months = [
        "[Jj]an(uary)?",
        "[Ff]eb(ruary)?",
        "[Mm]ar(ch)?",
        "[Mm]ay",
        "[Jj]un(e)?",
        "[Jj]ul(y)?",
        "[Aa]ug(ust)?",
        "[Ss]ep(tember)?",
        "[Oo]ct(ober)?",
        "[Nn]ov(ember)?",
        "[Dd]ec(ember)?",
        "JAN",
        "FEB",
        "MAR",
        "MAY",
        "JUN",
        "JUL",
        "AUG",
        "NOV",
        "DEC"
    ]
    # look for days in the week, and relative days (today, tomorrow, yesterday)
    for day in days:
        input_string = re.sub(day, "DATE", input_string)
    
    # look for MM/DD/YYYY, or MM/DD/YY, add or don't add zero at the beginning of numbers smaller than 10, didn't consider illegal numbers
    # also consider ancient years with 1,2 or 3 digits in years
    # didn't add AD/BC/BCE to the years
    input_string = re.sub("([0-1])?[0-9]\/([0-3])?[0-9]\/[0-9]{1,4}", "DATE", input_string)
    # look for DD/MM/YYYY, or DD/MM/YY, add or don't add zero at the beginning of numbers smaller than 10, didn't consider illegal numbers
    input_string = re.sub("([0-3])?[0-9]\/([0-1])?[0-9]\/[0-9]{1,4}", "DATE", input_string)
    # look for MM-DD-YYYY or DD-MM-YYYY
    input_string = re.sub("([0-1])?[0-9]\-([0-3])?[0-9]\-[0-9]{1,4}", "DATE", input_string)
    input_string = re.sub("([0-3])?[0-9]\-([0-1])?[0-9]\-[0-9]{1,4}", "DATE", input_string)
    # look for "Month DD, YYYY", "DD Month, YYYY", "YYYY, Month DD", "Month-DD-YYYY", "DD-Month-YYYY"
    for m in months:
        input_string = re.sub(m+" ([0-3])?[0-9]\, [0-9]{1,4}", "DATE", input_string)
        input_string = re.sub("([0-3])?[0-9] "+m+"\, [0-9]{1,4}", "DATE", input_string)
        input_string = re.sub("[0-9]{1,4}\, "+m+" ([0-3])?[0-9]", "DATE", input_string)
        input_string = re.sub(m+"\-([0-3])?[0-9]\-[0-9]{1,4}", "DATE", input_string)
        input_string = re.sub("([0-3])?[0-9]\-"+m+"\-[0-9]{1,4}", "DATE", input_string)
    # am or pm
    timeind = [
        "am",
        "pm",
        "AM",
        "PM"
    ]
    # 12 hour way with am/pm: "HH" or "HH:MM" or "HH:MM:SS" + am/pm
    for ind in timeind:
        input_string = re.sub("([0-1])?[0-9]:[0-5][0-9](:[0-5][0-9])? "+ind, "TIME", input_string)
        input_string = re.sub("([0-1])?[0-9] "+ind, "TIME", input_string)
    # 24 hour way: "HH:MM" or "HH:MM:SS"
    input_string = re.sub("([0-2])?[0-9]:[0-5][0-9](:[0-5][0-9])?", "TIME", input_string)   

    # 'o'clock' way 12 hour or 24 hour
    input_string = re.sub("([0-2])?[0-9] o\'clock", "TIME", input_string)
    
    

    return input_string # Feel free to modify this line if necessary

# GRADING: We will be importing and repeatedly calling your ner function from a separate script with various test case strings. For example (not exact):
# str1 = ner('she spent $149.99 and bought a nice microphone from www.bestdevices.com yesterday')
# if str1 == 'she spent DOLLAR_AMOUNT and bought a nice microphone from WEB_ADDRESS DATE':
#    correct = True
# print(correct)