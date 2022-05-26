# Item-Item Based Collaborative Filtering Recommender
import pandas as pd
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neighbors import NearestNeighbors


def main():
    print('Item-Item Based Collaborative Filtering Recommender: \n')
    # Data selection and processing
    dframe = pd.read_json('reviews_Health_and_Personal_Care_5.json', lines=True)

    # Drop the columns that I am not using in this algorithm
    dframe = dframe.drop(
        columns=['reviewerName', 'helpful', 'reviewText', 'unixReviewTime', 'reviewTime'])

    # Grouping by reviewerID to split users' ratings 80 training / 20 testing
    dframe_groupById = dframe.groupby(dframe.reviewerID)

    train_data = dframe_groupById.sample(frac=0.8, random_state=1)

    test_data = dframe.drop(train_data.index)

    test_data = test_data.pivot_table(values=['overall'], index=['reviewerID'], columns='asin')

    test_data_user_items = {}
    for i, row in test_data.iterrows():
        rows = [x for x in range(0, len(test_data.columns))]
        combine = list(zip(row.index, row.values, rows))
        rated = [(x, z) for x, y, z in combine if str(y) != 'nan']
        row_names = [i[0] for i in rated]
        test_data_user_items[i] = row_names

    training_pivot_rating = train_data.pivot_table(values=['overall'], index=['reviewerID'], columns='asin')

    user_items_rated = {}

    rows_indexes = {}
    for i, row in training_pivot_rating.iterrows():
        rows = [x for x in range(0, len(training_pivot_rating.columns))]
        combine = list(zip(row.index, row.values, rows))
        rated = [(x, z) for x, y, z in combine if str(y) != 'nan']
        index = [i[1] for i in rated]
        row_names = [i[0] for i in rated]
        rows_indexes[i] = index
        user_items_rated[i] = row_names
    # User item table
    pivot_table = train_data.pivot_table(values=['overall'], index=['reviewerID'], columns='asin').fillna(0)
    pivot_table = pivot_table.apply(np.sign)
    print(pivot_table)
    # Nearest Neighbor Recommender
    n = 5
    cosine_nn = NearestNeighbors(n_neighbors=n, algorithm='brute', metric='cosine')
    item_cosine_nn_fit = cosine_nn.fit(pivot_table.T.values)
    # item indices give the index of the actual item

    item_distances, item_indices = item_cosine_nn_fit.kneighbors(pivot_table.T.values)
    # The Predictions
    item_distances = 1 - item_distances
    predictions = item_distances.T.dot(pivot_table.T.values) / np.array([np.abs(item_distances.T).sum(axis=1)]).T
    ground_truth = pivot_table.T.values[item_distances.argsort()[0]]

    # Eval for predictions
    # Mean Absolute Error (MAE)
    def mae(prediction, ground_truth):
        prediction = prediction[ground_truth.nonzero()].flatten()
        ground_truth = ground_truth[ground_truth.nonzero()].flatten()
        return mean_absolute_error(prediction, ground_truth)

    # Root Mean Square Error
    def rmse(prediction, ground_truth):
        prediction = prediction[ground_truth.nonzero()].flatten()
        ground_truth = ground_truth[ground_truth.nonzero()].flatten()
        return sqrt(mean_squared_error(prediction, ground_truth))

    print("MAE: {:.2f}".format(mae(predictions, ground_truth)))
    print("RMSE: {:.2f}".format(rmse(predictions, ground_truth)))

    # Item Based Recommender
    items = {}
    print(range(len(pivot_table.T.index)))
    for i in range(len(pivot_table.T.index)):
        item_index = item_indices[i]
        col_names = pivot_table.T.index[item_index].tolist()
        items[pivot_table.T.index[i]] = col_names

    # Top recommendations_list
    top_recommendations_list = {}
    for userID, idx in rows_indexes.items():
        # indices for rated items
        item_index = [j for i in item_indices[idx] for j in i]
        # distances for rated items
        item_dist = [j for i in item_distances[idx] for j in i]
        combine = list(zip(item_dist, item_index))
        # filtering out the items that have been rated
        dictionary = {i: d for d, i in combine if i not in idx}
        zip_items = list(zip(dictionary.keys(), dictionary.values()))
        # similarity scores from the most similar to the least similar
        sort = sorted(zip_items, key=lambda x: x[1])
        recommendations_list = [(pivot_table.columns[i], d) for i, d in sort]
        top_recommendations_list[userID] = recommendations_list

    # Making a 10-item list of recommendations_list to all users
    def recommendations(user):
        to_file = '\n\n' + user + ' - ' + str(user_items_rated[user])
        users_recommendations.write(to_file)
        count = 0
        for a, b in top_recommendations_list.items():
            if user == a:
                for j in b[:10]:
                    to_file = '\n{} with similarity: {:.4f}'.format(j[0], 1 - j[1])
                    users_recommendations.write(to_file)
                    # Check if training data item in list of testing list items
                    if j[0] in test_data_user_items.get(user):
                        count += 1
        items_count = len(test_data_user_items.get(user))
        return count / 10, count / items_count

    users = np.unique(train_data.reviewerID.to_numpy())
    users_recommendations = open('recommendations_for_users.txt', 'a')
    users_counter = 0
    sum_precision = 0
    sum_recall = 0
    conversion_rate = 0
    for user in users:
        users_counter += 1
        count_precision, count_recall = recommendations_list(user)
        sum_precision += count_precision
        sum_recall += count_recall
        if count_precision > 0:
            conversion_rate += 1
    users_recommendations.close()

    # Eval for recommendations_list
    # Precision
    precision = (sum_precision / users_counter) * 100
    # Recall
    recall = (sum_recall / users_counter) * 100
    # F measure
    f_measure = 2 * precision * recall / (precision + recall)
    # Conversion rate
    conversion_rate = (conversion_rate / users_counter) * 100
    print('Precision= {:.2f}%'.format(precision))
    print('Recall= {:.2f}%'.format(recall))
    print('F-measure= {:.2f}'.format(f_measure))
    print('Conversion rate= {:.2f}%'.format(conversion_rate))


if __name__ == "__main__":
    main()
