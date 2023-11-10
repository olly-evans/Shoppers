import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4

def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """

    evidence = []
    labels = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)

        # Skip column names row.
        next(csv_reader)

        # For row in iterable, add evidence and labels.
        for row in csv_reader:
            # Int, float, int, float, int, float, float, float, float, float, 0-11 for month, int, int, int, int, 1 or 0, 1 or 0.
            converted_row = [
                int(row[0]),
                float(row[1]),
                int(row[2]),
                float(row[3]),
                int(row[4]),
                float(row[5]),
                float(row[6]),
                float(row[7]),
                float(row[8]),
                float(row[9]),
                month_to_int(row[10]),
                int(row[11]),
                int(row[12]),
                int(row[13]),
                int(row[14]),
                visitor_type(row[15]),
                bool_to_binary(row[16]),
            ]

            # append converted list/row to evidence, append label to labels.
            evidence.append(converted_row)
            labels.append(bool_to_binary(row[17]))
        #print(evidence[0])
        #print(labels[0])
    return (evidence, labels)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    # Create a k-nearest-neighbor classifier with k = 1
    knn_classifier = KNeighborsClassifier(n_neighbors=1)

    # Train the classifier on the provided data
    knn_classifier.fit(evidence, labels)

    return knn_classifier

def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """

    pos_total = 0
    neg_total = 0

    pos_correct = 0
    neg_correct = 0
    for i in range(len(labels)):
        if labels[i] == 1:
            pos_total += 1
        else:
            neg_total += 1

        if labels[i] == 1 and predictions[i] == 1:
            pos_correct += 1

        elif labels[i] == 0 and predictions[i] == 0:
            neg_correct += 1

    return (pos_correct/pos_total, neg_correct/neg_total)



def month_to_int(string):
    month_mapping = {
        'jan': 0, 'feb': 1, 'mar': 2, 'apr': 3,
        'may': 4, 'june': 5, 'jul': 6, 'aug': 7,
        'sep': 8, 'oct': 9, 'nov': 10, 'dec': 11
    }

    # Standardise/make lowercase.
    month = string.lower()
    
    # Return the associated int in the dict.
    return month_mapping.get(month, -1)

def bool_to_binary(boolean):
    if boolean == "TRUE":
        return 1
    else:
        return 0

def visitor_type(string):
    if string == "New_Visitor":
        return 0
    else:
        return 1

if __name__ == "__main__":
    main()
