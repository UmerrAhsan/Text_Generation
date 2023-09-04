import matplotlib.pyplot as plt



def print_max_min_length(df):

    df['length'] = df['text'].str.len()

    maximum_length = df['text'].str.len().max()
    print("Maximum length:", maximum_length)

    minimum_length = df['text'].str.len().min()
    print("Minimum length:", minimum_length)




def plot_histograms_for_individual_lengths(df):

    df['length'] = df['text'].str.len()

    # create a histogram of the string lengths
    plt.hist(df['length'], bins=50, align='left', edgecolor='black')
    plt.xlabel('String length')
    plt.ylabel('Frequency')
    plt.title('Distribution of string lengths')
    plt.show()

