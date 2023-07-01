#for testing case we come with separate dataset and test embeddigs 

#test embeddings file where you can also view train and test embeddings in above files which uploaded

test=np.load("/kaggle/input/t5embeds/test_embeds.npy")
print(test.shape)

test.shape[1]
test_df=pd.DataFrame(test,columns=["Column"+str(i) for i in range(1,test.shape[1]+1)])
print(test.shape,"\n")
test_df.head(3)
#evaluate the model and generate predictions

predictions=model.predict(test_df)
test_ids = np.load('/kaggle/input/t5embeds/test_ids.npy')

