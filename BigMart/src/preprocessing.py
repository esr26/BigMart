import pandas as pd

def preprocess(df):

    # Handling Null values

    # Fill Item_Weight
    df['Item_Weight'] = df.groupby('Item_Type')['Item_Weight'].transform(lambda x: x.fillna(x.median()))

    # Fill Outlet_Size
    df['Outlet_Size'] = df.groupby('Outlet_Type')['Outlet_Size'].transform(lambda x: x.fillna(x.mode()[0]))


    # Clean Item_Fat_Content
    df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({
        'LF': 'Low Fat',
        'low fat': 'Low Fat',
        'reg': 'Regular'
    })

    # Item_Category
    df['Item_Category'] = df['Item_Identifier'].str[:2]
    df['Item_Category'] = df['Item_Category'].map({
        'FD': 'Food',
        'NC': 'Non-Consumable',
        'DR': 'Drinks'
    })

    # Item_Age
    df['Item_Age'] = 2013 - df['Outlet_Establishment_Year']

    # Visibility Fix
    df['Item_Visibility'] = df.groupby('Item_Type')['Item_Visibility'].transform(
        lambda x: x.replace(0, x.mean())
    )

    df['Visibility_MeanRatio'] = df['Item_Visibility'] / df.groupby('Item_Type')['Item_Visibility'].transform('mean')

    # MRP Band
    df['MRP_Band'] = pd.cut(df['Item_MRP'], bins=4, labels=False)

    # Drop unused
    df = df.drop(['Outlet_Establishment_Year', 'Item_Identifier'], axis=1)

    return df
