#include "packed_depth.hpp"
#include "packed_schedule.hpp"
#include "packed_optimization.hpp"
#include <fstream>
#include "vector_chebyshev.hpp"
#include <iostream>
using namespace fhemamba;
// Cost-model diagnostic only. Actual CKKS levels and measured GPU timings are
// the adoption gates; this model cannot substitute for a full encrypted run.
int precise_cost(const PackedNode& n) {
  const int width=n.polynomials.size(), degree=n.polynomials[0].data.size()-3;
  std::vector<double> c((degree+1)*width);
  for(int j=0;j<width;++j)for(int k=0;k<=degree;++k)c[k*width+j]=n.polynomials[j].data[k+2];
  struct Ops {
    int multiply(int a,int b){return std::max(a,b)+1;}
    int add(int a,int b){return std::max(a,b);}
    int subtract(int a,int b){return std::max(a,b);}
    int scalar_add(int a,double){return a;}
    int scale(int a,double){return a+1;}
    int coefficient_product(int a,const std::vector<double>&){return a+1;}
    int constant(const std::vector<double>&){return 0;}
  } ops;
  return 1+evaluate_vector_chebyshev(0,width,degree,{c},ops,1e-14)[0];
}
void simulate(const PackedProgram& p, bool frontier, bool precise=false) {
  auto d=plan_packed_depth(p);
  if(precise)for(int i=0;i<(int)p.nodes.size();++i)if(p.nodes[i].operation=="cheb_batch")d.cost[i]=precise_cost(p.nodes[i]);
  if(frontier)std::fill(d.refresh_after.begin(),d.refresh_after.end(),false);
  PackedReadySchedule schedule(p,d.live);
  const auto uses=plan_packed_uses(p,d.live);
  std::vector<int> levels(p.nodes.size(),-1);
  std::vector<bool> present(p.nodes.size());
  int physical=0,logical=0,max_batch=1,max_ready=0,deferrals=0;
  auto refresh=[&](int request){
    std::vector<int> batch{request};int size=p.nodes[request].size;
    for(int j=0;j<(int)p.nodes.size() && batch.size()<16;++j){
      if(j==request||!present[j]||size+p.nodes[j].size>p.slots)continue;
      int next=schedule.next_consumer(j);
      if(next<0||p.nodes[next].operation=="feedback")continue;
      if(levels[j]<=d.refreshed||(levels[j]<33&&levels[j]+d.cost[next]<=39))continue;
      batch.push_back(j);size+=p.nodes[j].size;
    }
    for(int j:batch)levels[j]=d.refreshed;
    logical+=batch.size();++physical;max_batch=std::max(max_batch,(int)batch.size());
  };
  while(!schedule.empty()){
    max_ready=std::max(max_ready,(int)schedule.ready().size());
    int i=*schedule.ready().begin();
    if(frontier)for(int j:schedule.ready()){
      bool fits=true;
      if(p.nodes[j].operation!="feedback")for(int a:p.nodes[j].parents)fits &= levels[a]+d.cost[j]<=39;
      if(fits){if(i!=j)++deferrals;i=j;break;}
    }
    int level=-1;
    if(p.nodes[i].operation!="feedback")for(int a:p.nodes[i].parents){
      if(levels[a]+d.cost[i]>39)refresh(a);
      level=std::max(level,levels[a]);
    }
    levels[i]=level+d.cost[i];present[i]=true;
    schedule.complete(i);
    if(d.refresh_after[i]&&levels[i]>d.refreshed)refresh(i);
    for(int a:p.nodes[i].parents)if(schedule.releasable(a))present[a]=false;
  }
  std::cout<<"{\"frontier\":"<<(frontier?"true":"false")<<",\"precise_polynomial_depth\":"<<(precise?"true":"false")<<",\"logical_refreshes\":"<<logical<<",\"physical_refreshes\":"<<physical<<",\"max_batch\":"<<max_batch<<",\"max_ready\":"<<max_ready<<",\"deferrals\":"<<deferrals<<"}";
}
int main(int argc,char**argv){std::ifstream f(argv[1]);auto p=read_packed_program(f,true);
PackedOptimizationStats stats;auto b=batch_packed_polynomials(std::move(p),stats);
int count=0,total=0;for(const auto&n:b.nodes)if(n.operation=="cheb_batch"){int difference=packed_node_depth(n)-precise_cost(n);if(difference)++count;total+=difference;}
std::cout<<"{\"batches_with_depth_reduction\":"<<count<<",\"total_saved_depth\":"<<total<<",\"simulations\":[";
simulate(b,true,false);std::cout<<',';simulate(b,true,true);std::cout<<"]\n";}
