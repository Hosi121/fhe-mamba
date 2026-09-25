#include "packed_depth.hpp"
#include "packed_schedule.hpp"
#include "packed_optimization.hpp"
#include <fstream>
#include <iostream>
using namespace fhemamba;
// Cost-model diagnostic only. Actual CKKS levels and measured GPU timings are
// the adoption gates; this model cannot substitute for a full encrypted run.
void simulate(const PackedProgram& p, bool frontier) {
  auto d=plan_packed_depth(p);
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
  std::cout<<"{\"frontier\":"<<(frontier?"true":"false")<<",\"logical_refreshes\":"<<logical<<",\"physical_refreshes\":"<<physical<<",\"max_batch\":"<<max_batch<<",\"max_ready\":"<<max_ready<<",\"deferrals\":"<<deferrals<<"}";
}
int main(int argc,char**argv){std::ifstream f(argv[1]);auto p=read_packed_program(f,true);
std::cout<<"{\"scope\":\"static level model, not measurement\",\"original\":[";simulate(p,false);std::cout<<',';simulate(p,true);
PackedOptimizationStats stats;auto b=batch_packed_polynomials(std::move(p),stats);
std::cout<<"],\"batched\":[";simulate(b,false);std::cout<<',';simulate(b,true);std::cout<<"]}\n";}
