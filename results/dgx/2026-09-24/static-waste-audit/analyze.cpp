// Source-level call inventory only. No encryption, model inference or timing.
#include "packed_depth.hpp"
#include "packed_routing.hpp"
#include "stage1_mamba2_plan.hpp"
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <set>
#include <tuple>

struct ChebCost {
  long long ctct=0, scalar=0, adds=0, zero_adds=0, constant_leaves=0, zero_leaves=0;
  long long constant_upper=0, owned_leaf_adds=0, recombines=0, basis_double=0;
  int depth=0;
};
ChebCost cheb_cost(const std::vector<double>& coefficients, int baby) {
  ChebCost cost;
  std::map<int,int> cache{{1,0}};
  std::function<int(int)> basis = [&](int i) -> int {
    if (cache.count(i)) return cache.at(i);
    int level;
    if (i%2==0) {level=basis(i/2)+1;++cost.adds;}
    else {const auto high=basis((i+1)/2),low=basis(i/2);level=std::max(high,low)+1;++cost.adds;}
    ++cost.ctct;++cost.basis_double;++cost.adds;
    return cache[i]=level;
  };
  struct Value {int level;bool constant;double value;};
  std::function<Value(std::vector<double>)> evaluate = [&](std::vector<double> c) -> Value {
    int degree=static_cast<int>(c.size())-1;
    if (degree<baby) {
      int terms=0,level=0;
      for (int i=1;i<=degree;++i) if (std::abs(c[i])>=1e-14) {
        ++terms;++cost.scalar;level=std::max(level,basis(i)+1);
      }
      if (!terms) {++cost.scalar;++cost.constant_leaves;cost.zero_leaves+=c[0]==0;level=1;}
      else {cost.adds+=terms-1;cost.owned_leaf_adds+=terms-1;}
      ++cost.adds;cost.zero_adds+=c[0]==0;
      return {level,terms==0,c[0]};
    }
    int k=baby;while(2*k-1<degree)k*=2;
    std::vector<double> upper(c.begin()+k,c.end()),lower(c.begin(),c.begin()+k);
    for(int i=1;i<static_cast<int>(upper.size());++i)upper[i]*=2;
    for(int i=k+1;i<=degree;++i)lower[2*k-i]-=c[i];
    auto left=evaluate(lower),right=evaluate(upper);
    const int giant=basis(k);
    ++cost.ctct;++cost.adds;++cost.recombines;cost.constant_upper+=right.constant;
    return {std::max(left.level,std::max(giant,right.level)+1),false,0};
  };
  cost.depth=evaluate(coefficients).level;
  return cost;
}

struct Routing {
  long long calls=0,stages=0,masks=0,adds=0,bsgs_stages=0;
  std::set<std::pair<bool,std::vector<double>>> geometries;
  void add(const std::vector<double>& indices,bool scatter,int slots) {
    ++calls;geometries.insert({scatter,indices});
    bool monotone=true,identity=true;std::set<int> offsets;
    for(int i=0;i<static_cast<int>(indices.size());++i){
      monotone=monotone&&(!i||indices[i]>indices[i-1]);identity=identity&&indices[i]==i;
      offsets.insert((scatter?-1:1)*(static_cast<int>(indices[i])-i));
    }
    if(scatter&&identity)return;
    if(monotone&&offsets.size()>32){
      for(const auto& stage:fhemamba::monotone_routing(indices,scatter)){
        ++stages;masks+=stage.size();adds+=stage.size()-1;
        std::vector<int> shifts;for(const auto& [k,v]:stage)shifts.push_back(k);
        bsgs_stages+=fhemamba::plan_packed_diagonals(shifts,slots,true).baby_step!=0;
      }
    }else{masks+=offsets.size();adds+=offsets.size()-1;}
  }
};

int main(int argc,char**argv){
  if(argc!=2)return 2;
  std::ifstream in(argv[1]);const auto p=fhemamba::read_packed_program(in,true);
  const auto plan=fhemamba::plan_packed_depth(p);
  Routing routing;
  long long linears=0,diagonals=0,linear_inner_adds=0,linear_outer_adds=0;
  long long fill_fold_adds=0,chebs=0,dag_muls=0,zero_adds=0,ctct=0,scalar=0;
  long long constant_leaves=0,zero_leaves=0,constant_upper=0,owned_leaf_adds=0,recombines=0;
  long long tuned_ctct=0,tuned_scalar=0,tuned_changed=0;
  std::map<int,std::pair<int,int>> weights;
  std::map<std::vector<double>,int> poly_uses;
  std::map<std::tuple<int,int,int,int,int>,int> shapes;
  std::cout<<"{\"scope\":\"Static source call inventory; model masks only, excluding refresh interiors; no runtime speed claim\",\"cheb_plans\":[";
  bool first=true;
  for(int index=0;index<static_cast<int>(p.nodes.size());++index){
    if(!plan.live[index])continue;
    const auto& n=p.nodes[index];dag_muls+=n.operation=="mul";
    if(n.operation=="linear"||n.operation=="linear_ref"){
      ++linears;const int columns=p.nodes[n.parents[0]].size;
      const auto shape=fhemamba::stage1::resolve_interleaved_replicated_shape(n.size,columns,p.slots,0);
      if(shape.replicas<=1)throw std::runtime_error("Unmodeled dense fallback");
      const int baby=std::max(2,static_cast<int>(std::sqrt(shape.per_replica)));
      const int giants=(shape.per_replica+baby-1)/baby;
      diagonals+=shape.per_replica;
      linear_inner_adds+=shape.per_replica-giants;linear_outer_adds+=giants-1;
      fill_fold_adds+=fhemamba::stage1::rotation_sum_schedule(shape.reps,true).size()+
        fhemamba::stage1::rotation_sum_schedule(shape.replicas+shape.guard_windows,true).size()+
        fhemamba::stage1::rotation_sum_schedule(shape.replicas,true).size();
      const int id=n.operation=="linear"?index:static_cast<int>(n.data[0]);
      ++weights[id].first;weights[id].second=shape.per_replica;
      ++shapes[{n.size,columns,shape.replicas,shape.per_replica,baby}];
    }else if(n.operation=="cheb"){
      ++chebs;std::vector<double> c(n.data.begin()+2,n.data.end());double at_zero=0;
      for(int i=0;i<static_cast<int>(c.size());i+=2)at_zero+=(i%4?-1:1)*c[i];
      c[0]-=at_zero;++poly_uses[c];int levels=0;while((1<<levels)<static_cast<int>(c.size()))++levels;
      const int baby=1<<((std::max(1,levels)+1)/2);
      auto cost=cheb_cost(c,baby),best=cost;int best_baby=baby;
      for(int b=2;b<=static_cast<int>(c.size());++b){
        auto trial=cheb_cost(c,b);
        if(trial.depth<=cost.depth&&std::tie(trial.ctct,trial.scalar)<std::tie(best.ctct,best.scalar)){
          best=trial;best_baby=b;
        }
      }
      ctct+=cost.ctct;scalar+=cost.scalar;zero_adds+=cost.zero_adds;
      constant_leaves+=cost.constant_leaves;zero_leaves+=cost.zero_leaves;constant_upper+=cost.constant_upper;
      owned_leaf_adds+=cost.owned_leaf_adds;recombines+=cost.recombines;
      tuned_ctct+=best.ctct;tuned_scalar+=best.scalar;tuned_changed+=baby!=best_baby;
      if(baby!=best_baby){
        if(!first)std::cout<<',';first=false;
        std::cout<<"{\"node\":"<<index<<",\"degree\":"<<c.size()-1<<",\"baby\":"<<baby
                 <<",\"candidate_baby\":"<<best_baby<<",\"ctct\":"<<cost.ctct<<",\"candidate_ctct\":"<<best.ctct
                 <<",\"scalar\":"<<cost.scalar<<",\"candidate_scalar\":"<<best.scalar
                 <<",\"depth\":"<<cost.depth<<",\"candidate_depth\":"<<best.depth<<'}';
      }
    }
    if(n.operation=="gather"||n.operation=="scatter")routing.add(n.data,n.operation=="scatter",p.slots);
    else if(n.operation=="repeat"){
      int outer=n.data[0],inner=n.data[1],repeat=n.data[2];std::vector<double> indices(outer*inner);
      for(int g=0;g<outer;++g)for(int j=0;j<inner;++j)indices[g*inner+j]=g*inner*repeat+j;
      routing.add(indices,true,p.slots);
    }else if(n.operation=="sum"){
      std::vector<double> indices(n.size);for(int j=0;j<n.size;++j)indices[j]=j*n.data[0];routing.add(indices,false,p.slots);
    }
  }
  long long unique_diagonals=0;for(const auto& [id,value]:weights)unique_diagonals+=value.second;
  std::cout<<"],\"linears\":{\"calls\":"<<linears<<",\"unique_weights\":"<<weights.size()
           <<",\"diagonal_encodes\":"<<diagonals<<",\"unique_diagonal_identities\":"<<unique_diagonals
           <<",\"inner_adds\":"<<linear_inner_adds<<",\"outer_adds\":"<<linear_outer_adds
           <<",\"fill_fold_adds\":"<<fill_fold_adds<<",\"shapes\":[";
  first=true;for(const auto& [shape,count]:shapes){
    if(!first)std::cout<<',';first=false;auto [rows,columns,replicas,diags,baby]=shape;
    std::cout<<"{\"rows\":"<<rows<<",\"columns\":"<<columns<<",\"replicas\":"<<replicas
             <<",\"diagonals\":"<<diags<<",\"baby\":"<<baby<<",\"calls\":"<<count<<'}';
  }
  std::cout<<"]},\"routing\":{\"calls\":"<<routing.calls<<",\"unique_geometries\":"<<routing.geometries.size()
           <<",\"radix_stages\":"<<routing.stages<<",\"optimized_stages\":"<<routing.bsgs_stages
           <<",\"mask_encodes\":"<<routing.masks<<",\"accumulator_adds\":"<<routing.adds<<"},\"chebyshev\":{\"calls\":"<<chebs
           <<",\"unique_adjusted_polynomials\":"<<poly_uses.size()<<",\"ctct\":"<<ctct<<",\"scalar_multiplies\":"<<scalar
           <<",\"zero_scalar_adds\":"<<zero_adds<<",\"constant_leaves\":"<<constant_leaves<<",\"zero_leaves\":"<<zero_leaves
           <<",\"constant_upper_children\":"<<constant_upper<<",\"owned_leaf_adds\":"<<owned_leaf_adds
           <<",\"recombines\":"<<recombines<<",\"tuned_ctct\":"<<tuned_ctct<<",\"tuned_scalar\":"<<tuned_scalar
           <<",\"tuned_calls\":"<<tuned_changed<<"},\"model_ctct_from_static_calls\":"<<dag_muls+ctct<<"}\n";
}
