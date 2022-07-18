def xml2tables(path):
    import xml.etree.ElementTree as ET
    import pandas as pd
    tree = ET.parse(data)
    key = []
    value = []
    for i in tree.iter():
        print(str(i))
        key.append(str(i).split("'")[1])
        print(i.text)
        value.append(i.text)
        print("*"*120)
    df = pd.DataFrame({"Key":key, "Values":value})
    df2 = df[df["Values"]!="\n"]
    return df2