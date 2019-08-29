#include <DataCollectionHandle.h>
#include <ExtremumGraph.h>

using namespace HDFileFormat;

int main(){
    DataCollectionHandle collection;// = DataCollectionHandle();
    collection.attach("summaryTopologyTest.hdff");

    DatasetHandle &dataset = collection.dataset(0);
    DataBlockHandle &datablock = dataset.getDataBlock(0);

    ExtremumGraphExt eg;

    eg.load(datablock);
    
    return 0;
}
