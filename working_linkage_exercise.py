'''
Data Linkage: Working-Level Course (part 2)

We have two datasets which we want to link: working_data_a and working_data_b

 - working_data_a contains the variables: id_a, firstname, middlename, surname, sex, dob, 
postcode and a record ID that is contained in the variable: ident_b. In addition 
there is a variable, ident_a,  that contains the record ID from the small file 
that it is matched to, i.e. we know the true match status.

 - working_data_b contains the variables: id_b, firstname, middlename, surname, sex, dob, 
postcode and a record ID that is contained in the variable: ident_a. In addition 
there is a variable, ident_b,  that contains the record ID from the large file 
that it is matched to, i.e. we know the true match status.
'''

# Import pandas for data manipulation, numpy for filtering, os to read the working
# directory and re for regular expressions
import pandas as pd
import numpy as np
import re
import os

pd.set_option('display.max_columns', None)

def get_file_path():
    # This is the filepath where the datasets and the matchkey file can be found
    file_path = os.getcwd()
    return file_path


def read_data(filepath):
    # Widen output display
    pd.set_option('display.width', 1000)
    # Read in datasets to link
    dfA = pd.read_csv(filepath + '/working_data_a.csv')
    dfB = pd.read_csv(filepath + '/working_data_b.csv')
    # Make sure column types correct
    dfA = dfA.astype({"id_a":str, "firstname_a":str, "middlename_a":str, "surname_a":str,
                      "sex_a":str, "dob_a":str, "postcode_a": str})
    
    dfB = dfB.astype({"id_b":str, "firstname_b":str, "middlename_b":str, "surname_b":str,
                      "sex_b":str, "dob_b":str, "postcode_b": str})
    # Drop the ident_a and ident_b columns as we don't need them
    return (dfA.drop(['ident_a','ident_b'], axis=1), dfB.drop(['ident_a','ident_b'], axis=1))


def clean_data(df, letter):
    # Convert standardise the name variables
    df = standardise_names(df, letter)
    # Standardise the postcodes
    df = standardise_postcode(df, 'postcode_' + letter)
    # Standardise the sex
    df = standardise_sex(df, 'sex_' + letter)
    # Standardise the date of birth
    df = standardise_dob(df, 'dob_' + letter)
    return df


def standardise_names(df, letter):
    for name in add_subscript(letter, ["firstname", "middlename", "surname"]):
        # Ensure all names are uppercase
        df[name] = df[name].str.upper()
        # Ensure names contain no whitespace at the beginning or end, only have 
        # single spaces internally and convert all hyphens to spaces
        df[name] = df[name].str.replace('-',' ').str.replace('  ',' ').str.strip()
        # Replace any empty names or the name "NAN" with 'None'
        df[name] = np.where((df[name] == 'NAN') | (df[name] == ''), None, df[name])
        # Convert the column type to str
        df.astype({name:str}, copy=False)
    # Split each name into two variables on the delimiter ' '. Later name variables
    # will be 'None' if there are not two names in that column. Titles will be 
    # stripped off into a title column
    return split_names(df, letter)


def split_names(df, letter):  
    # Split firstname. As it may contain a title and two names, we need to temporarily
    # create three firstname variables (the last of which will be dropped later)
    f1, f2, f3 = add_subscript(letter, ['first1', 'first2','first3'])
    df[[f1, f2, f3]] = df['firstname_'+letter].str.split(' ', n=2, expand=True)
    # Remove any titles by putting them in a separate title column
    titles = ['MR', 'MRS', 'MISS', 'MS', 'DR']
    df['title_'+letter] = np.where(df[f1].isin(titles), df[f1], None)
    # If firstname1 is a title, swap the order of the forename variables so it
    # becomes forename3 (the column we will drop)
    df[f1], df[f2] = np.where(df[f1].isin(titles), [df[f2],df[f1]], [df[f1],df[f2]])
    df[f2], df[f3] = np.where(df[f2].isin(titles), [df[f3],df[f2]], [df[f2],df[f3]])
    # Split middlename
    df[add_subscript(letter, ['middle1', 'middle2'])] = df['middlename_'+letter].str.split(' ', n=1, expand=True)
    # Split surname
    df['sur1_'+letter], df['sur2_'+letter] = df['surname_'+letter], None
    #df[add_subscript(letter, ['sur1', 'sur2'])] = df['surname_'+letter].str.split(' ', n=1, expand=True)
    # Firstname3 no longer needed as it will only contain titles
    return df.drop(f3, axis=1)


def standardise_postcode(df, pc):
    # Remove all white space from the postcodes and make them uppercase
    df[pc] = df[pc].str.replace(' ','').str.upper()
    # Replace missing postcodes with 'None'
    df[pc] = np.where((df[pc] == 'NAN') | (df[pc] == '') | (df[pc].isnull()), None, df[pc])
    # Convert the type to string
    return df.astype({pc:str})


def standardise_sex(df, sex):
    # Convert the sex to uppercase
    df[sex] = df[sex].str.upper()
    # Change 'M' and 'MALE' to 1 and 'F' and 'FEMALE' to 2
    df.loc[(df[sex] == 'M') | (df[sex] == 'MALE'), sex] = 1
    df.loc[(df[sex] == 'F') | (df[sex] == 'FEMALE'), sex] = 2
    # Convert the type to a float
    return df.astype({sex:float})


def standardise_dob(df, dob):
    df[dob] = pd.to_datetime(df[dob], dayfirst=True)
    return df


def create_derived_variables(df, letter):
    # Make new columns containing only the first three letters and initials of each
    # part of each name. Also create a column for initials
    df = short_names(df, letter)
    # Create day, month and year variables for each dataset
    dob = 'dob_' + letter
    df['daybirth_'+letter] = df[dob].dt.day
    df['monthbirth_'+letter] = df[dob].dt.month
    df['yearbirth_'+letter] = df[dob].dt.year
    # Separate out the different parts of the postcode
    return split_postcode(df, letter)
   
    
def short_names(df, letter):
    # Create columns containing the first three letters and initial of each name
    f1, f2, m1, m2, s1, s2 = add_subscript(letter, ['first1', 'first2', 'middle1', 'middle2', 'sur1', 'sur2'])
    for name in [f1, f2, m1, m2, s1, s2]:
        df['short'+name] = df[name].str[:3]
        df['init'+name] = df[name].str[0]
    # Combine all of the initial columns to give a person's initials
    df['initials_'+letter] = df[['init'+f1,'init'+f2,'init'+m1,'init'+m2,'init'+s1,'init'+s2]].apply(
            lambda row: row.str.cat(sep=''), axis=1)
    return df


def split_postcode(df, letter):
    # Split out the postcode area and district
    pc, area, district = add_subscript(letter, ['postcode', 'pcarea', 'pcdistrict'])
    df[area] = df[pc].str.extract('\A([A-Z]{1,2})', expand=True)
    df[district] = df[pc].str.extract('\A([A-Z]{1,2}[0-9]{1,2})[0-9]', expand=True)
    df[district] = np.where((df[district].isnull()) & (df[pc].str.match('[A-Z]{1,2}[0-9]{1,2}')),
                              df[pc], df[district])
    return df


def add_subscript(letter, args):
    # Add the letter as a subscript to each variable
    args_subscript = []
    for arg in args:
        args_subscript.append(str(arg) + '_' + letter)
    return args_subscript


def remove_subscript(args):
    # Remove the subscript from each variable
    args_no_subscript = []
    for arg in args:
        args_no_subscript.append(str(arg).split('_')[0])
    return args_no_subscript


def exact_matching(dfA, dfB):
    # Exact matching is just rule-based matching on all of the columns
    return rule_based_matching(dfA, dfB, remove_subscript(dfA.columns.values), False)


def create_matchkeys(filepath):
    # Read in the matchkeys file
    file = open(filepath + '/working_matchkeys.txt')
    lines = file.read().splitlines()
    file.close()
    matchkeys = []
    keep_multiple = False
    for line in lines:
        if re.match('Include multiple matches', line):
            # Set the value of keep_multiple to True or False as appropriate. If neither
            # True or False was inputted, raise a value error.
            value = re.search('Include multiple matches *=(.*)', line).group(1).replace(' ','')
            if value.upper() == 'TRUE':
                keep_multiple = True
            elif value.replace(' ','').upper() == 'FALSE':
                keep_multiple = False
            else:
                raise ValueError('"Include multiple matches" in the file matchkeys.txt '+\
                                 'must be either "true" or "false" but received "' + value + '"')
        elif line == '':
            # Do nothing as it's a blank line
            continue
        else:
            # Add the matchkey to the list of matchkeys
            matchkeys.append(line.replace(' ','').split(','))
    return keep_multiple, matchkeys


def rule_based_matching(dfA, dfB, matchkey, keep_multiple):
    # Create the variable names for dfA and dfB
    left = add_subscript('a', matchkey)
    right = add_subscript('b', matchkey)
    # Join on the columns given by args
    linked = dfA.merge(dfB, left_on = left, right_on = right, how = 'inner')
    multiple_matches = None
    if keep_multiple == False:
        # Remove the multiple matches
        linked['multi_match'] = linked['id_a'].map(linked['id_a'].value_counts()>1)\
                                | linked['id_b'].map(linked['id_b'].value_counts()>1)
        multiple_matches = len(linked[linked['multi_match']])
        linked = linked[linked['multi_match'] == False].drop('multi_match', axis=1)
    # Add a column 'Match_Status' that has value 1 if the match is correct and 0 otherwise
    linked['Match_Status'] = np.where(linked['id_a'] == linked['id_b'],1,0)
    true_positives = len(linked[linked['Match_Status']==1])
    false_positives = len(linked[linked['Match_Status']==0])
    # Find the residuals
    residuals = dfA.merge(dfB, left_on = left, right_on = right, how = 'outer')
    # Calculate residuals from dfA. We want to include all records from dfA that
    # are not in linked but we don't want to include any records from dfB (as
    # these would just be a row of null values)
    residualsA = residuals[residuals['id_a'].notnull() &
                           ~residuals['id_a'].isin(linked['id_a'])][dfA.columns.values]\
                           .drop_duplicates()
    # Calculate residuals from dfB. We want to include all records from dfB that
    # are not in linked but we don't want to include any records from dfA (as
    # these would just be a row of null values)
    residualsB = residuals[residuals['id_b'].notnull() & 
                           ~residuals['id_b'].isin(linked['id_b'])][dfB.columns.values]\
                           .drop_duplicates()
    unmatched = len(residualsB)
    # Print information on the number of true positives, false positives and 
    # unmatched records for this matchkey
    print_match_rate(False, left == dfA.columns.values.tolist(), matchkey,
                     multiple_matches, true_positives, false_positives, unmatched, None)
    # Update dfA and dfB to remove the matched records
    dfA = residualsA
    dfB = residualsB
    return (dfA, dfB, (true_positives, false_positives))


def print_match_rate(final, exact, matchkey, multiple_matches, true_positives,
                     false_positives, unmatched, false_negatives):
    # Print details of the type of matching and the number of true positives, false
    # positives etc. in a readable form
    if final:
        print('Overall match rate')
    elif exact:
        print('Exact matching')
    else:
        print('Matchkey = ' + ', '.join(matchkey))
        if multiple_matches != None:
            print('Multiple matches = ' + str(multiple_matches))
    print('True positives = ' + str(true_positives))
    print('False positives = ' + str(false_positives))
    if unmatched != None:
        print('Unmatched = ' + str(unmatched))
    if false_negatives != None:
        print('False negatives = ' + str(false_negatives))
    print('')


def update_match_rate(true_positives, false_positives, link_status):
    # Update the number of true positives and false positives found
    true_positives += link_status[0]
    false_positives += link_status[1]
    return(true_positives, false_positives)


def get_precision(true_positives, false_positives):
    # Calculate the precision as a percentage to 2 decimal places
    return round(true_positives/(true_positives+false_positives)*100, 2)


def get_recall(true_positives, false_negatives):
    # Calculate the recall as a percentage to 2 decimal places
    return round(true_positives/(true_positives+false_negatives)*100, 2)

# Get the filepath from the command line arguments
filepath = get_file_path()

# Read in the data, clean it and create any derived variables so it is ready for matching
dfA, dfB = read_data(filepath)
dfA = clean_data(dfA, "a")
dfB = clean_data(dfB, "b")
dfA = create_derived_variables(dfA, "a")
dfB = create_derived_variables(dfB, "b")
   
# Set the number of false negatives to be the length of dataset B and the number 
# of true positives and false positives to be zero as we haven't made any matches
false_neg = len(dfB)
true_pos, false_pos = 0, 0
    
# Run the exact matching and update the number of true positives and false positives
dfA, dfB, link_status = exact_matching(dfA, dfB)
true_pos, false_pos = update_match_rate(true_pos, false_pos, link_status)
    
# Read in the list of matchkeys and convert each to a list of variables on which to match
keep_multiple, matchkeys = create_matchkeys(filepath)
    
# For each matchkey, run the rule-based matching and then update the number of 
# true positives and false positives
for matchkey in matchkeys:
    dfA, dfB, link_status = rule_based_matching(dfA, dfB, matchkey, keep_multiple)
    true_pos, false_pos = update_match_rate(true_pos, false_pos, link_status)
    
# Calculate the number of false negatives
false_neg -= true_pos
    
# Print overall summary information and calculate the precision and recall
print_match_rate(True, False, None, None, true_pos, false_pos, None, false_neg)
print('Precision = ' + str(get_precision(true_pos, false_pos)) + '%')
print('Recall = ' + str(get_recall(true_pos, false_neg)) + '%')