from pymongo import MongoClient
import pymongo

class MongoDB:
    """
    Create the mongodb client

    Attributes
    ----------
        IP : str
            Binding an IP address
        port : int
            MongoDB Port number (Default : 27017)
        collection : str
            The name of the database collection which we want to use

    Methods
    -------
        addItem(name, embeddings)
            Adding a new item to the database collection
        getDictonary()
            Get database entries as a dictionary
        getItems()
            Get database entries as array
        getNameByID(ID)
            Get the name of the item by its ID 
    """
    def __init__(self, IP, port, collection):
        """
        Parameters
        ----------
            IP : str
                Binding an IP address
            port : int
                MongoDB Port number (Default : 27017)
            collection : str
                The name of the database collection which we want to use
        """
        self.client = MongoClient(IP, port)
        self.db = self.client.UI
        self.collection = self.db[collection]

    def addItem(self, name, embeddings):
        """Adding a new item to the database collection

        Parameters
        ----------
            name : str
                The name of new item
            embeddings : list[float32]
                The array of embeddings
        """
        cur = self.collection.find()   
        results = list(cur)
        if len(results)==0:
            ID = 'P'+hex(1)
        else:
            last = int(results[len(results)-1]['_id'][1:],16)
            ID = 'P'+hex(last+1)

        item = {
        '_id' : ID,
        'Name' : name,
        'Emb' : embeddings}
        self.collection.insert_one(item)

    def getDictonary(self):
        """Get database entries as a dictionary
        
        Returns:
        -------
            dict : dictionary
                The dictionary contains all the information from the database collection
        """
        cur = self.collection.find()   
        dict = list(cur)
        return dict

    def getItems(self):
        """Get database entries as array

        Returns:
        -------
            dbIDs : list[str]
                List of items ID
            dbLabels : list[str]
                List of items Name
            dbEmbs : list[float32]
                List of items Embedding
        """
        cur = self.collection.find()   
        items = list(cur)
        dbIDs = []
        dbLabels = []
        dbEmbs = []        
        for db_item in items:
            for db_item_emb in db_item['Emb']:
                dbIDs.append(db_item['_id'])
                dbLabels.append(db_item['Name'])
                dbEmbs.append(db_item_emb)
        return dbIDs, dbLabels, dbEmbs

    def getNameByID(self,ID):
        """Get the name of the item by its ID 

        Parameters
        ----------
            ID : str
                The ID of the item you are looking for

        Returns:
        -------
            item : dictionary
                Searched item
        """
        assert bool(ID)
        item = self.collection.find_one({"_id":ID})
        return item
