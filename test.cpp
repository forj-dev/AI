#include<bits/stdc++.h>

using namespace std;

class Model;
class Layer;
class AdversarialModel;
class Linear;
class ReLU;
class Sigmoid;
class SoftMax;
double MSELoss(const vector<double>& out,const vector<double>& target);
double CrossEntropyLoss(const vector<double>& out, const vector<double>& target);


//Utils

template<typename T>
vector<T> &operator+(vector<T> &v1,vector<T>& v2){
	vector<T> v(v1);
	v.insert(v.end(),v2.begin(),v2.end());
	return v1;
}

int maxProbability(const vector<double>& v){
	int res=0;
	for(int i=1;i<v.size();i++){
		if(v[i]>v[res])res=i;
	}
	return res;
}

void printVector(const vector<double>& v){
	for(double val:v){
		cout<<val<<" ";
	}
	cout<<endl<<endl;
}

void printMatrix(const vector< vector<double> > &m){
	for(auto& v:m){
		for(double val:v){
			cout<<val<<" ";
		}
		cout<<endl;
	}
	cout<<endl;
}


//Models

class Layer{
public:
	virtual const void registerArgs(Model* model){}
	virtual vector<double> forward(const vector<double>& in){return vector<double>();}
};


const double dx=1e-10;

class Model{
public:
	vector<double*> args;
	double (*loss)(const vector<double>& out,const vector<double>& target);
	
	Model(double (*lossFunction)(const vector<double>& out,const vector<double>& target)){
		loss=lossFunction;
	}
	
	void addArg(double* arg){
		args.emplace_back(arg);
	}
	
	void registerLayers(vector<Layer*>& layers){
		for(Layer* layer:layers){
			layer->registerArgs(this);
		}
	}
	
	virtual vector<double> forward(const vector<double>& in){return vector<double>();}
	
	void backward(const vector<double>& in,const vector<double>& target,const double speed){
		double orginLoss=(*loss)(forward(in),target);
		vector<double> losses(args.size());
		double lossSqDis=0;
		for(int i=0;i<args.size();i++){
			double orginArg=*(args[i]);
			*(args[i])+=dx;
			losses[i]=(*loss)(forward(in),target);
			lossSqDis+=losses[i]*losses[i];
			*(args[i])=orginArg;
		}
		double multiplier=speed*orginLoss/dx/sqrt(lossSqDis);
		for(int i=0;i<args.size();i++){
			*(args[i])-=(losses[i]-orginLoss)*multiplier;
		}
	}
};


class AdversarialModel{
public:
	int inputRandomCount,inputCount,outputCount;
	Model *generator,*discriminator;
	AdversarialModel(int _inputRandomCount,int _inputCount,int _outputCount,
			Model *_generator,Model *_discriminator){
		inputRandomCount=_inputRandomCount;
		inputCount=_inputCount;
		outputCount=_outputCount;
		generator=_generator;
		discriminator=_discriminator;
		generator->loss=
	}
	void backward(vector<double>& in,vector<double>& randIn,vector<double>& target,
			const double speed){
		const vector<double> firstLossTarget({1.0,0.0}),secondLossTarget({0.0,1.0});
		vector<double> fullGenIn=in+randIn;
		vector<double> generated=generator->forward(fullGenIn);
		bool realInput=rand()&1;//false: gen+real true:real+gen
		vector<double> fullDisIn=(realInput?target+generated:generated+target);
		vector<double> probabilities=discriminator->forward(fullDisIn);
		double generatorLoss=CrossEntropyLoss(probabilities,
			(realInput?secondLossTarget:firstLossTarget)
		);
		double discriminatorLoss=CrossEntropyLoss(probabilities,
			(realInput?firstLossTarget:secondLossTarget)
		);
	}
};


//Layers

class Linear:public Layer{
public:
	vector< vector<double> > weights;
	vector<double> bias;
	Linear(int inSize,int outSize,double initialWeight,double initialBias,double randomRange){
		weights=vector<vector<double>>(inSize,vector<double>(outSize,initialWeight));
		bias=vector<double>(outSize,initialBias);
		if(randomRange>0){
			for(auto &vec:weights)
				for(double &val:vec)
					val+=randomRange*((double)rand()/RAND_MAX*2-1);
			for(double &val:bias)
				val+=randomRange*((double)rand()/RAND_MAX*2-1);
		}
	}
	const void registerArgs(Model* model)override{
		for(auto &vec:weights)
			for(double &val:vec)
				model->addArg(&val);
		for(double &val:bias)
			model->addArg(&val);
	}
	vector<double> forward(const vector<double>& in)override{
		if(in.size()!=weights.size()){
			cout<<"LinearLayer: Input and weights vectors must be of the same size: "
				<<in.size()<<" , "<<weights.size()<<endl;
			throw invalid_argument("LinearLayer: Input and weights vectors must be of the same size.");
		}
		vector<double> out(bias);
		for(int i=0;i<weights.size();i++)
			for(int j=0;j<weights[0].size();j++)
				out[j]+=in[i]*weights[i][j];
		return out;
	}
};

class ReLU:public Layer{
public:
	vector<double> forward(const vector<double>& in)override{
		vector<double> out(in.size());
		for(int i=0;i<in.size();i++)out[i]=max(0.0,in[i]);
		return out;
	}
};

class Sigmoid:public Layer{
public:
	vector<double> forward(const vector<double>& in)override{
		vector<double> out(in.size());
		for(int i=0;i<in.size();i++)out[i]=1/(1+exp(-in[i]));
		return out;
	}
};

class SoftMax:public Layer{
public:
	vector<double> forward(const vector<double>& in)override{
    	double sum=accumulate(in.begin(),in.end(),0.0,
			[](double acc,double x){
				return acc+exp(x);
			}
		);
		if(sum==0){
			return vector<double>(in);
		}
    	vector<double> out(in.size());
    	for(int i=0;i<in.size();i++){
    	    out[i]=exp(in[i])/sum;
    	}
    	return out;
	}
};


//LossFunctions

double MSELoss(const vector<double>& out,const vector<double>& target){
	if(out.size()!=target.size()){
		cout<<"MSELoss: Input and target vectors must be of the same size: "
			<<out.size()<<" , "<<target.size()<<endl;
		throw invalid_argument("MSELoss: Input and target vectors must be of the same size.");
	}
	double loss=0;
	for(int i=0;i<out.size();i++)
		loss+=pow(out[i]-target[i],2);
	return loss;
}

double CrossEntropyLoss(const vector<double>& out, const vector<double>& target) {
	const double _LOSS_INF_=1e10;
    if(out.size()!=target.size()){
		cout<<"CrossEntropyLoss: Input and target vectors must be of the same size: "
			<<out.size()<<" , "<<target.size()<<endl;
		throw invalid_argument("CrossEntropyLoss: Input and target vectors must be of the same size.");
	}
    double loss=0;
    for(int i=0;i<out.size();i++){
    	if(out[i]<0||out[i]>1){
			cout<<"CrossEntropyLoss: Input probabilities must be between 0 and 1: ["
				<<i<<"] "<<out[i]<<endl;
    		throw invalid_argument("CrossEntropyLoss: Input probabilities must be between 0 and 1.");
    	}
        if(target[i]==1.0){
        	if(out[i]==0)loss+=_LOSS_INF_;
            else loss-=log(out[i]);
		}
    }
    return loss;
}



//Test

/*
class TestModel:public Model{
public:
	Layer* l1;
	TestModel():Model(MSELoss){
		srand(time(0));
		l1=new Linear(1,1,0,0,0.5);
		vector<Layer*> layers({l1});
		registerLayers(layers);
	}
	~TestModel(){
		delete l1;
	}
	vector<double> forward(const vector<double>& in)override{
		return l1->forward(in);
	}
};

int main(){
	Model* model=new TestModel();
	vector< vector<double> > train_in({{0},{1},{2},{3},{4},{5}});
	vector< vector<double> > train_target({{0.1},{0.95},{2.12},{3},{3.97},{5.21}});
	vector< vector<double> > test_in({{0},{1},{2},{3},{4},{5},{6},{7},{8},{9}});
	for(int i=0;i<100;i++){
		for(int j=0;j<train_in.size();j++){
			model->backward(train_in[j],train_target[j],0.01);
		}
	}
	cout<<*(model->args[0])<<" "<<*(model->args[1])<<endl;
	for(int i=0;i<test_in.size();i++){
		cout<<model->forward(test_in[i])[0]<<" ";
	}
	delete(model);
	return 0;
}
//*/

//*
class TestModel:public Model{
public:
	Linear *l1,*l2;
	ReLU *relu;
	Sigmoid *sg;
	SoftMax *sm;
	TestModel():Model(CrossEntropyLoss){
		srand(time(0));
		l1=new Linear(2,2,0,0,0.5);
		l2=new Linear(2,2,0,0,0.5);
		relu=new ReLU();
		sg=new Sigmoid();
		sm=new SoftMax();
		vector<Layer*> layers({l1,l2,relu,sm});
		registerLayers(layers);
	}
	~TestModel(){
		delete l1;
		delete l2;
		delete relu;
		delete sm;
	}
	vector<double> forward(const vector<double>& in)override{
		auto v=l1->forward(in);
		v=sg->forward(v);
		v=l2->forward(v);
		return sm->forward(v);
	}
};

int main(){
	TestModel* model=new TestModel();
	vector< vector<double> > train_in({
		{-1.9,3.9},
		{-0.7,1.9},
		{1.9,3.7},
		{1.1,3.2},
		{-0.3,4.3},
		{2.4,-2.5},
		{-2.3,0.3},
		{1.1,-1.6},
		{0.1,-3.4},
		{-1.5,-2.7},
		{-0.7,-0.7},
		{-3.8,1.6},
		{3.1,0.9},
		{2.3,-0.5}
	});
	vector< vector<double> > train_target({
		{1.0,0.0},
		{1.0,0.0},
		{1.0,0.0},
		{1.0,0.0},
		{1.0,0.0},
		{0.0,1.0},
		{0.0,1.0},
		{0.0,1.0},
		{0.0,1.0},
		{0.0,1.0},
		{0.0,1.0},
		{0.0,1.0},
		{0.0,1.0},
		{0.0,1.0},
	});
	vector< vector<double> > test_in({
		{-0.3,-0.5},
		{1.2,3.5},
		{-3.6,0.2},
		{-1.2,3.5}
	});
	for(int i=0;i<1000;i++){
		for(int j=0;j<train_in.size();j++){
			model->backward(train_in[j],train_target[j],0.01);
		}
	}
	for(int i=0;i<test_in.size();i++){
		cout<<maxProbability(model->forward(test_in[i]))<<" ";
	}
	cout<<endl;
	printMatrix(model->l1->weights);
	printVector(model->l1->bias);
	printMatrix(model->l2->weights);
	printVector(model->l2->bias);
	delete(model);
	return 0;
}
//*/







