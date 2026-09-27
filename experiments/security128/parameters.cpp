// No keys or GPU allocations: audit actual generated Q/P and rejection paths.
#include "fideslib_security.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
using namespace fideslib;

static std::string json_string(const std::string& value) {
  std::string result="\"";
  for (char c:value) {
    if (c=='\\' || c=='\"') { result+='\\'; result+=c; }
    else if (c=='\n') result+="\\n";
    else if (c=='\r') result+="\\r";
    else if (c=='\t') result+="\\t";
    else result+=c;
  }
  return result+'\"';
}

int main(int argc,char** argv) {
  if (argc!=2) return 2;
  try {
    std::ostringstream rows;
    int accepted=0,rejected=0;
    bool first=true;
    for (int ring : {65536,131072}) for (int digits : {3,4,6}) {
      if (!first) rows<<',';
      first=false;
      rows<<"{\"requested_ring\":"<<ring<<",\"digits\":"<<digits;
      bool pass=false;
      try {
        CCParams<CryptoContextCKKSRNS> p;
        p.SetSecurityLevel(HEStd_128_classic); p.SetSecretKeyDist(UNIFORM_TERNARY);
        p.SetCKKSDataType(REAL); p.SetRingDim(ring); p.SetBatchSize(ring/2);
        p.SetMultiplicativeDepth(44); p.SetScalingModSize(59); p.SetFirstModSize(60);
        p.SetScalingTechnique(FLEXIBLEAUTO); p.SetKeySwitchTechnique(HYBRID);
        p.SetNumLargeDigits(digits);
        auto c=GenCryptoContext(p);
        auto audit=fhemamba::audit_ckks_context(c);
        rows<<",\"audit\":";audit.write_json(rows);
        audit.require_classical128();pass=true;
      } catch(const std::exception& e) { rows<<",\"rejection\":"<<json_string(e.what()); }
      rows<<",\"accepted\":"<<(pass?"true":"false")<<'}';
      if (pass) ++accepted; else ++rejected;
      if (pass != (ring==131072 && digits>=4))
        throw std::runtime_error("unexpected secure parameter feasibility");
    }
    // The audit must reject an explicitly experimental context, even if its
    // ring and moduli happen to fit the numeric bound.
    for (bool sparse : {false,true}) {
      CCParams<CryptoContextCKKSRNS> p;
      p.SetSecurityLevel(sparse ? HEStd_128_classic : HEStd_NotSet);
      p.SetSecretKeyDist(sparse ? SPARSE_TERNARY : UNIFORM_TERNARY);
      p.SetRingDim(131072);p.SetBatchSize(65536);p.SetMultiplicativeDepth(44);
      p.SetScalingModSize(59);p.SetFirstModSize(60);p.SetNumLargeDigits(4);
      p.SetScalingTechnique(FLEXIBLEAUTO);p.SetKeySwitchTechnique(HYBRID);
      auto c=GenCryptoContext(p);auto a=fhemamba::audit_ckks_context(c);
      if(a.satisfies_classical128())throw std::runtime_error("unsupported security assumptions accepted");
      bool refused=false;
      try { a.require_classical128(); } catch(const std::invalid_argument&) { refused=true; }
      if(!refused)throw std::runtime_error("fail-closed guard did not reject");
    }
    std::ofstream out(argv[1]);
    out<<"{\"passed\":true,\"keys_generated\":0,\"gpu_allocation_telemetry\":false,\"accepted\":"<<accepted
       <<",\"rejected\":"<<rejected<<",\"assumption_rejection_checks\":2,\"cases\":["<<rows.str()<<"]}\n";
    std::cout<<"parameter audit passed accepted="<<accepted<<" rejected="<<rejected<<'\n';
  } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
