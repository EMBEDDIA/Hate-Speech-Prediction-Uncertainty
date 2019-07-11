## example tests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap

def get_embedding(dfile):

    nf = pd.DataFrame()
    embedding = umap.UMAP().fit_transform(dfile.iloc[:,1:])
    correctness = dfile['correct'].values
    nf['d1'] = embedding[:,0]
    nf['d2'] = embedding[:,1]
    nf['labels'] = dfile['correct']
    nf['predictions'] = dfile.iloc[:,1:].mean(axis=1).round(0)
    nf['Probability range'] = pd.cut(dfile.iloc[:,1:].mean(axis=1),4)
    nf['Probability'] = dfile.iloc[:,1:].mean(axis=1)
    return nf

def plot_basic(df,text_df):

    df = pd.merge(df, text_df, left_index=True, right_index=True)
    df['real class'] = np.random.randint(2,size=df.shape[0])
    df['Prediction vs. real'] = "Correct: "+df['labels'].astype(str) + ", Predicted: " +df['predictions'].astype(str)
    sizes = df.Probability*700    
    sns.scatterplot(df.d1,df.d2,hue=df['Probability range'],style=df['Prediction vs. real'],s=sizes,palette=["blue","green","orange","red"])
    sns.kdeplot(df.d1,df.d2,color="green",alpha=0.1,shade=False)
    for idx, row in df.iterrows():
        plt.text(row['d1']+0.05,row['d2']+0.05,row['index'],fontsize=15)

    plt.xlabel("Latent dimension 1")
    plt.ylabel("Latent dimension 2")
    plt.axis('off')
    plt.show()

    subframe = df[['index','text']]
    pd.options.display.max_colwidth = 500
#    print(subframe)
    subframe.to_latex("./result_images/tweets.tex",index=False)

def plot_kde(df):

    df['preds'] = df.iloc[:,1:].mean(axis=1).round(0)
    df['correct'] = np.where(df['preds']==df['correct'], 1, 0)
    grouped = df.groupby(['correct','preds']).mean().reset_index()
    grouped['means'] = df.iloc[:,1:].mean(axis=1)
    sns.kdeplot(grouped.means,grouped.correct,color="black",shade=True,label="Correct")
    sns.kdeplot(grouped.means,grouped.preds,color="red",label="Predicted")
    plt.xlabel("Mean probability")
    plt.ylabel("Prediction value")
    plt.ylim(top=1,bottom=0)
    plt.xlim(left=1,right=0)
    plt.legend()
    plt.show()
    
if __name__ == "__main__":

    dfile = "./data/data_5.csv"
    text_file = "./data/test_5.csv"
    texts = pd.read_csv(text_file,sep=",")
    texts.columns = ["index","instance id","text"]
    dmat = np.genfromtxt(dfile,delimiter="\t").T
    names = ["correct"]+[str(ex) for ex, v in enumerate(range(dmat.shape[1]-1))]
    dataframe = pd.DataFrame.from_records(dmat)
    dataframe.columns = names
    emb_df = get_embedding(dataframe)
    plot_basic(emb_df,texts)
    
#    plot_kde(dataframe)
