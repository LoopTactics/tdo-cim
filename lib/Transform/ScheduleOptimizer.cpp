//===- Schedule.cpp - Calculate an optimized schedule ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass generates an entirely new schedule tree from the data dependences
// and iteration domains. The new schedule tree is computed in two steps:
//
// 1) The isl scheduling optimizer is run
//
// The isl scheduling optimizer creates a new schedule tree that maximizes
// parallelism and tileability and minimizes data-dependence distances. The
// algorithm used is a modified version of the ``Pluto'' algorithm:
//
//   U. Bondhugula, A. Hartono, J. Ramanujam, and P. Sadayappan.
//   A Practical Automatic Polyhedral Parallelizer and Locality Optimizer.
//   In Proceedings of the 2008 ACM SIGPLAN Conference On Programming Language
//   Design and Implementation, PLDI ’08, pages 101–113. ACM, 2008.
//
// 2) A set of post-scheduling transformations is applied on the schedule tree.
//
// These optimizations include:
//
//  - Tiling of the innermost tilable bands
//  - Prevectorization - The choice of a possible outer loop that is strip-mined
//                       to the innermost level to enable inner-loop
//                       vectorization.
//  - Some optimizations for spatial locality are also planned.
//
// For a detailed description of the schedule tree itself please see section 6
// of:
//
// Polyhedral AST generation is more than scanning polyhedra
// Tobias Grosser, Sven Verdoolaege, Albert Cohen
// ACM Transactions on Programming Languages and Systems (TOPLAS),
// 37(4), July 2015
// http://www.grosser.es/#pub-polyhedral-AST-generation
//
// This publication also contains a detailed discussion of the different options
// for polyhedral loop unrolling, full/partial tile separation and other uses
// of the schedule tree.
//
//===----------------------------------------------------------------------===//

#include "polly/ScheduleOptimizer.h"
#include "polly/CodeGen/CodeGeneration.h"
#include "polly/DependenceInfo.h"
#include "polly/LinkAllPasses.h"
#include "polly/Options.h"
#include "polly/ScopInfo.h"
#include "polly/ScopPass.h"
#include "polly/Simplify.h"
#include "polly/Support/GICHelper.h"
#include "polly/Support/ISLOStream.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "isl/constraint.h"
#include "isl/ctx.h"
#include "isl/map.h"
#include "isl/options.h"
#include "isl/printer.h"
#include "isl/schedule.h"
#include "isl/schedule_node.h"
#include "isl/space.h"
#include "isl/union_map.h"
#include "isl/union_set.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "polly/Access.h"
#include "polly/Access_patterns.h"
#include "polly/Builders.h"
#include "polly/Matchers.h"
#include <regex>

using namespace llvm;
using namespace polly;

#define DEBUG_TYPE "polly-opt-isl"

static cl::opt<std::string>
    OptimizeDeps("polly-opt-optimize-only",
                 cl::desc("Only a certain kind of dependences (all/raw)"),
                 cl::Hidden, cl::init("all"), cl::ZeroOrMore,
                 cl::cat(PollyCategory));

static cl::opt<std::string>
    SimplifyDeps("polly-opt-simplify-deps",
                 cl::desc("Dependences should be simplified (yes/no)"),
                 cl::Hidden, cl::init("yes"), cl::ZeroOrMore,
                 cl::cat(PollyCategory));

static cl::opt<int> MaxConstantTerm(
    "polly-opt-max-constant-term",
    cl::desc("The maximal constant term allowed (-1 is unlimited)"), cl::Hidden,
    cl::init(20), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<int> MaxCoefficient(
    "polly-opt-max-coefficient",
    cl::desc("The maximal coefficient allowed (-1 is unlimited)"), cl::Hidden,
    cl::init(20), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<std::string> FusionStrategy(
    "polly-opt-fusion", cl::desc("The fusion strategy to choose (min/max)"),
    cl::Hidden, cl::init("min"), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<std::string>
    MaximizeBandDepth("polly-opt-maximize-bands",
                      cl::desc("Maximize the band depth (yes/no)"), cl::Hidden,
                      cl::init("yes"), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<std::string> OuterCoincidence(
    "polly-opt-outer-coincidence",
    cl::desc("Try to construct schedules where the outer member of each band "
             "satisfies the coincidence constraints (yes/no)"),
    cl::Hidden, cl::init("no"), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<int> PrevectorWidth(
    "polly-prevect-width",
    cl::desc(
        "The number of loop iterations to strip-mine for pre-vectorization"),
    cl::Hidden, cl::init(4), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<bool> FirstLevelTiling("polly-tiling",
                                      cl::desc("Enable loop tiling"),
                                      cl::init(true), cl::ZeroOrMore,
                                      cl::cat(PollyCategory));

static cl::opt<int> LatencyVectorFma(
    "polly-target-latency-vector-fma",
    cl::desc("The minimal number of cycles between issuing two "
             "dependent consecutive vector fused multiply-add "
             "instructions."),
    cl::Hidden, cl::init(8), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<int> ThroughputVectorFma(
    "polly-target-throughput-vector-fma",
    cl::desc("A throughput of the processor floating-point arithmetic units "
             "expressed in the number of vector fused multiply-add "
             "instructions per clock cycle."),
    cl::Hidden, cl::init(1), cl::ZeroOrMore, cl::cat(PollyCategory));

// This option, along with --polly-target-2nd-cache-level-associativity,
// --polly-target-1st-cache-level-size, and --polly-target-2st-cache-level-size
// represent the parameters of the target cache, which do not have typical
// values that can be used by default. However, to apply the pattern matching
// optimizations, we use the values of the parameters of Intel Core i7-3820
// SandyBridge in case the parameters are not specified or not provided by the
// TargetTransformInfo.
static cl::opt<int> FirstCacheLevelAssociativity(
    "polly-target-1st-cache-level-associativity",
    cl::desc("The associativity of the first cache level."), cl::Hidden,
    cl::init(-1), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<int> FirstCacheLevelDefaultAssociativity(
    "polly-target-1st-cache-level-default-associativity",
    cl::desc("The default associativity of the first cache level"
             " (if not enough were provided by the TargetTransformInfo)."),
    cl::Hidden, cl::init(8), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<int> SecondCacheLevelAssociativity(
    "polly-target-2nd-cache-level-associativity",
    cl::desc("The associativity of the second cache level."), cl::Hidden,
    cl::init(-1), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<int> SecondCacheLevelDefaultAssociativity(
    "polly-target-2nd-cache-level-default-associativity",
    cl::desc("The default associativity of the second cache level"
             " (if not enough were provided by the TargetTransformInfo)."),
    cl::Hidden, cl::init(8), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<int> FirstCacheLevelSize(
    "polly-target-1st-cache-level-size",
    cl::desc("The size of the first cache level specified in bytes."),
    cl::Hidden, cl::init(-1), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<int> FirstCacheLevelDefaultSize(
    "polly-target-1st-cache-level-default-size",
    cl::desc("The default size of the first cache level specified in bytes"
             " (if not enough were provided by the TargetTransformInfo)."),
    cl::Hidden, cl::init(32768), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<int> SecondCacheLevelSize(
    "polly-target-2nd-cache-level-size",
    cl::desc("The size of the second level specified in bytes."), cl::Hidden,
    cl::init(-1), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<int> SecondCacheLevelDefaultSize(
    "polly-target-2nd-cache-level-default-size",
    cl::desc("The default size of the second cache level specified in bytes"
             " (if not enough were provided by the TargetTransformInfo)."),
    cl::Hidden, cl::init(262144), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<int> VectorRegisterBitwidth(
    "polly-target-vector-register-bitwidth",
    cl::desc("The size in bits of a vector register (if not set, this "
             "information is taken from LLVM's target information."),
    cl::Hidden, cl::init(-1), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<int> FirstLevelDefaultTileSize(
    "polly-default-tile-size",
    cl::desc("The default tile size (if not enough were provided by"
             " --polly-tile-sizes)"),
    cl::Hidden, cl::init(32), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::list<int>
    FirstLevelTileSizes("polly-tile-sizes",
                        cl::desc("A tile size for each loop dimension, filled "
                                 "with --polly-default-tile-size"),
                        cl::Hidden, cl::ZeroOrMore, cl::CommaSeparated,
                        cl::cat(PollyCategory));

static cl::opt<bool>
    SecondLevelTiling("polly-2nd-level-tiling",
                      cl::desc("Enable a 2nd level loop of loop tiling"),
                      cl::init(false), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<int> SecondLevelDefaultTileSize(
    "polly-2nd-level-default-tile-size",
    cl::desc("The default 2nd-level tile size (if not enough were provided by"
             " --polly-2nd-level-tile-sizes)"),
    cl::Hidden, cl::init(16), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::list<int>
    SecondLevelTileSizes("polly-2nd-level-tile-sizes",
                         cl::desc("A tile size for each loop dimension, filled "
                                  "with --polly-default-tile-size"),
                         cl::Hidden, cl::ZeroOrMore, cl::CommaSeparated,
                         cl::cat(PollyCategory));

static cl::opt<bool> RegisterTiling("polly-register-tiling",
                                    cl::desc("Enable register tiling"),
                                    cl::init(false), cl::ZeroOrMore,
                                    cl::cat(PollyCategory));

static cl::opt<int> RegisterDefaultTileSize(
    "polly-register-tiling-default-tile-size",
    cl::desc("The default register tile size (if not enough were provided by"
             " --polly-register-tile-sizes)"),
    cl::Hidden, cl::init(2), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<int> PollyPatternMatchingNcQuotient(
    "polly-pattern-matching-nc-quotient",
    cl::desc("Quotient that is obtained by dividing Nc, the parameter of the"
             "macro-kernel, by Nr, the parameter of the micro-kernel"),
    cl::Hidden, cl::init(256), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::list<int>
    RegisterTileSizes("polly-register-tile-sizes",
                      cl::desc("A tile size for each loop dimension, filled "
                               "with --polly-register-tile-size"),
                      cl::Hidden, cl::ZeroOrMore, cl::CommaSeparated,
                      cl::cat(PollyCategory));

static cl::opt<bool>
    PMBasedOpts("polly-pattern-matching-based-opts",
                cl::desc("Perform optimizations based on pattern matching"),
                cl::init(true), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<bool> OptimizedScops(
    "polly-optimized-scops",
    cl::desc("Polly - Dump polyhedral description of Scops optimized with "
             "the isl scheduling optimizer and the set of post-scheduling "
             "transformations is applied on the schedule tree"),
    cl::init(false), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<bool> MatcherOptLate(
    "polly-enable-matchers-opt-late",
    cl::desc("Performs optimizations based on pattern matching (after isl)."),
    cl::init(false), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<bool> MatcherOptEarly(
    "polly-enable-matchers-opt-early",
    cl::desc("Performs optimizations based on pattern matching (before isl)."),
    cl::init(false), cl::ZeroOrMore, cl::cat(PollyCategory));

STATISTIC(ScopsProcessed, "Number of scops processed");
STATISTIC(ScopsRescheduled, "Number of scops rescheduled");
STATISTIC(ScopsOptimized, "Number of scops optimized");

STATISTIC(NumAffineLoopsOptimized, "Number of affine loops optimized");
STATISTIC(NumBoxedLoopsOptimized, "Number of boxed loops optimized");

#define THREE_STATISTICS(VARNAME, DESC)                                        \
  static Statistic VARNAME[3] = {                                              \
      {DEBUG_TYPE, #VARNAME "0", DESC " (original)", {0}, {false}},            \
      {DEBUG_TYPE, #VARNAME "1", DESC " (after scheduler)", {0}, {false}},     \
      {DEBUG_TYPE, #VARNAME "2", DESC " (after optimizer)", {0}, {false}}}

THREE_STATISTICS(NumBands, "Number of bands");
THREE_STATISTICS(NumBandMembers, "Number of band members");
THREE_STATISTICS(NumCoincident, "Number of coincident band members");
THREE_STATISTICS(NumPermutable, "Number of permutable bands");
THREE_STATISTICS(NumFilters, "Number of filter nodes");
THREE_STATISTICS(NumExtension, "Number of extension nodes");

STATISTIC(FirstLevelTileOpts, "Number of first level tiling applied");
STATISTIC(SecondLevelTileOpts, "Number of second level tiling applied");
STATISTIC(RegisterTileOpts, "Number of register tiling applied");
STATISTIC(PrevectOpts, "Number of strip-mining for prevectorization applied");
STATISTIC(MatMulOpts,
          "Number of matrix multiplication patterns detected and optimized");

constexpr int TILE_FACTOR_CIM_DEVICE = 256;

/// Create an isl::union_set, which describes the isolate option based on
/// IsolateDomain.
///
/// @param IsolateDomain An isl::set whose @p OutDimsNum last dimensions should
///                      belong to the current band node.
/// @param OutDimsNum    A number of dimensions that should belong to
///                      the current band node.
static isl::union_set getIsolateOptions(isl::set IsolateDomain,
                                        unsigned OutDimsNum) {
  unsigned Dims = IsolateDomain.dim(isl::dim::set);
  assert(OutDimsNum <= Dims &&
         "The isl::set IsolateDomain is used to describe the range of schedule "
         "dimensions values, which should be isolated. Consequently, the "
         "number of its dimensions should be greater than or equal to the "
         "number of the schedule dimensions.");
  isl::map IsolateRelation = isl::map::from_domain(IsolateDomain);
  IsolateRelation = IsolateRelation.move_dims(isl::dim::out, 0, isl::dim::in,
                                              Dims - OutDimsNum, OutDimsNum);
  isl::set IsolateOption = IsolateRelation.wrap();
  isl::id Id = isl::id::alloc(IsolateOption.get_ctx(), "isolate", nullptr);
  IsolateOption = IsolateOption.set_tuple_id(Id);
  return isl::union_set(IsolateOption);
}

namespace {
/// Create an isl::union_set, which describes the specified option for the
/// dimension of the current node.
///
/// @param Ctx    An isl::ctx, which is used to create the isl::union_set.
/// @param Option The name of the option.
isl::union_set getDimOptions(isl::ctx Ctx, const char *Option) {
  isl::space Space(Ctx, 0, 1);
  auto DimOption = isl::set::universe(Space);
  auto Id = isl::id::alloc(Ctx, Option, nullptr);
  DimOption = DimOption.set_tuple_id(Id);
  return isl::union_set(DimOption);
}
} // namespace

/// Create an isl::union_set, which describes the option of the form
/// [isolate[] -> unroll[x]].
///
/// @param Ctx An isl::ctx, which is used to create the isl::union_set.
static isl::union_set getUnrollIsolatedSetOptions(isl::ctx Ctx) {
  isl::space Space = isl::space(Ctx, 0, 0, 1);
  isl::map UnrollIsolatedSetOption = isl::map::universe(Space);
  isl::id DimInId = isl::id::alloc(Ctx, "isolate", nullptr);
  isl::id DimOutId = isl::id::alloc(Ctx, "unroll", nullptr);
  UnrollIsolatedSetOption =
      UnrollIsolatedSetOption.set_tuple_id(isl::dim::in, DimInId);
  UnrollIsolatedSetOption =
      UnrollIsolatedSetOption.set_tuple_id(isl::dim::out, DimOutId);
  return UnrollIsolatedSetOption.wrap();
}

/// Make the last dimension of Set to take values from 0 to VectorWidth - 1.
///
/// @param Set         A set, which should be modified.
/// @param VectorWidth A parameter, which determines the constraint.
static isl::set addExtentConstraints(isl::set Set, int VectorWidth) {
  unsigned Dims = Set.dim(isl::dim::set);
  isl::space Space = Set.get_space();
  isl::local_space LocalSpace = isl::local_space(Space);
  isl::constraint ExtConstr = isl::constraint::alloc_inequality(LocalSpace);
  ExtConstr = ExtConstr.set_constant_si(0);
  ExtConstr = ExtConstr.set_coefficient_si(isl::dim::set, Dims - 1, 1);
  Set = Set.add_constraint(ExtConstr);
  ExtConstr = isl::constraint::alloc_inequality(LocalSpace);
  ExtConstr = ExtConstr.set_constant_si(VectorWidth - 1);
  ExtConstr = ExtConstr.set_coefficient_si(isl::dim::set, Dims - 1, -1);
  return Set.add_constraint(ExtConstr);
}

isl::set getPartialTilePrefixes(isl::set ScheduleRange, int VectorWidth) {
  unsigned Dims = ScheduleRange.dim(isl::dim::set);
  isl::set LoopPrefixes =
      ScheduleRange.drop_constraints_involving_dims(isl::dim::set, Dims - 1, 1);
  auto ExtentPrefixes = addExtentConstraints(LoopPrefixes, VectorWidth);
  isl::set BadPrefixes = ExtentPrefixes.subtract(ScheduleRange);
  BadPrefixes = BadPrefixes.project_out(isl::dim::set, Dims - 1, 1);
  LoopPrefixes = LoopPrefixes.project_out(isl::dim::set, Dims - 1, 1);
  return LoopPrefixes.subtract(BadPrefixes);
}

isl::schedule_node
ScheduleTreeOptimizer::isolateFullPartialTiles(isl::schedule_node Node,
                                               int VectorWidth) {
  assert(isl_schedule_node_get_type(Node.get()) == isl_schedule_node_band);
  Node = Node.child(0).child(0);
  isl::union_map SchedRelUMap = Node.get_prefix_schedule_relation();
  isl::map ScheduleRelation = isl::map::from_union_map(SchedRelUMap);
  isl::set ScheduleRange = ScheduleRelation.range();
  isl::set IsolateDomain = getPartialTilePrefixes(ScheduleRange, VectorWidth);
  auto AtomicOption = getDimOptions(IsolateDomain.get_ctx(), "atomic");
  isl::union_set IsolateOption = getIsolateOptions(IsolateDomain, 1);
  Node = Node.parent().parent();
  isl::union_set Options = IsolateOption.unite(AtomicOption);
  Node = Node.band_set_ast_build_options(Options);
  return Node;
}

isl::schedule_node ScheduleTreeOptimizer::prevectSchedBand(
    isl::schedule_node Node, unsigned DimToVectorize, int VectorWidth) {
  assert(isl_schedule_node_get_type(Node.get()) == isl_schedule_node_band);

  auto Space = isl::manage(isl_schedule_node_band_get_space(Node.get()));
  auto ScheduleDimensions = Space.dim(isl::dim::set);
  assert(DimToVectorize < ScheduleDimensions);

  if (DimToVectorize > 0) {
    Node = isl::manage(
        isl_schedule_node_band_split(Node.release(), DimToVectorize));
    Node = Node.child(0);
  }
  if (DimToVectorize < ScheduleDimensions - 1)
    Node = isl::manage(isl_schedule_node_band_split(Node.release(), 1));
  Space = isl::manage(isl_schedule_node_band_get_space(Node.get()));
  auto Sizes = isl::multi_val::zero(Space);
  Sizes = Sizes.set_val(0, isl::val(Node.get_ctx(), VectorWidth));
  Node =
      isl::manage(isl_schedule_node_band_tile(Node.release(), Sizes.release()));
  Node = isolateFullPartialTiles(Node, VectorWidth);
  Node = Node.child(0);
  // Make sure the "trivially vectorizable loop" is not unrolled. Otherwise,
  // we will have troubles to match it in the backend.
  Node = Node.band_set_ast_build_options(
      isl::union_set(Node.get_ctx(), "{ unroll[x]: 1 = 0 }"));
  Node = isl::manage(isl_schedule_node_band_sink(Node.release()));
  Node = Node.child(0);
  if (isl_schedule_node_get_type(Node.get()) == isl_schedule_node_leaf)
    Node = Node.parent();
  auto LoopMarker = isl::id::alloc(Node.get_ctx(), "SIMD", nullptr);
  PrevectOpts++;
  return Node.insert_mark(LoopMarker);
}

isl::schedule_node ScheduleTreeOptimizer::tileNode(isl::schedule_node Node,
                                                   const char *Identifier,
                                                   ArrayRef<int> TileSizes,
                                                   int DefaultTileSize) {
  auto Space = isl::manage(isl_schedule_node_band_get_space(Node.get()));
  auto Dims = Space.dim(isl::dim::set);
  auto Sizes = isl::multi_val::zero(Space);
  std::string IdentifierString(Identifier);
  for (unsigned i = 0; i < Dims; i++) {
    auto tileSize = i < TileSizes.size() ? TileSizes[i] : DefaultTileSize;
    Sizes = Sizes.set_val(i, isl::val(Node.get_ctx(), tileSize));
  }
  auto TileLoopMarkerStr = IdentifierString + " - Tiles";
  auto TileLoopMarker =
      isl::id::alloc(Node.get_ctx(), TileLoopMarkerStr, nullptr);
  Node = Node.insert_mark(TileLoopMarker);
  Node = Node.child(0);
  Node =
      isl::manage(isl_schedule_node_band_tile(Node.release(), Sizes.release()));
  Node = Node.child(0);
  auto PointLoopMarkerStr = IdentifierString + " - Points";
  auto PointLoopMarker =
      isl::id::alloc(Node.get_ctx(), PointLoopMarkerStr, nullptr);
  Node = Node.insert_mark(PointLoopMarker);
  return Node.child(0);
}

isl::schedule_node ScheduleTreeOptimizer::applyRegisterTiling(
    isl::schedule_node Node, ArrayRef<int> TileSizes, int DefaultTileSize) {
  Node = tileNode(Node, "Register tiling", TileSizes, DefaultTileSize);
  auto Ctx = Node.get_ctx();
  return Node.band_set_ast_build_options(isl::union_set(Ctx, "{unroll[x]}"));
}

static bool isSimpleInnermostBand(const isl::schedule_node &Node) {
  assert(isl_schedule_node_get_type(Node.get()) == isl_schedule_node_band);
  assert(isl_schedule_node_n_children(Node.get()) == 1);

  auto ChildType = isl_schedule_node_get_type(Node.child(0).get());

  if (ChildType == isl_schedule_node_leaf)
    return true;

  if (ChildType != isl_schedule_node_sequence)
    return false;

  auto Sequence = Node.child(0);

  for (int c = 0, nc = isl_schedule_node_n_children(Sequence.get()); c < nc;
       ++c) {
    auto Child = Sequence.child(c);
    if (isl_schedule_node_get_type(Child.get()) != isl_schedule_node_filter)
      return false;
    if (isl_schedule_node_get_type(Child.child(0).get()) !=
        isl_schedule_node_leaf)
      return false;
  }
  return true;
}

bool ScheduleTreeOptimizer::isTileableBandNode(isl::schedule_node Node) {
  if (isl_schedule_node_get_type(Node.get()) != isl_schedule_node_band)
    return false;

  if (isl_schedule_node_n_children(Node.get()) != 1)
    return false;

  if (!isl_schedule_node_band_get_permutable(Node.get()))
    return false;

  auto Space = isl::manage(isl_schedule_node_band_get_space(Node.get()));
  auto Dims = Space.dim(isl::dim::set);

  if (Dims <= 1)
    return false;

  return isSimpleInnermostBand(Node);
}

__isl_give isl::schedule_node
ScheduleTreeOptimizer::standardBandOpts(isl::schedule_node Node, void *User) {
  if (FirstLevelTiling) {
    Node = tileNode(Node, "1st level tiling", FirstLevelTileSizes,
                    FirstLevelDefaultTileSize);
    FirstLevelTileOpts++;
  }

  if (SecondLevelTiling) {
    Node = tileNode(Node, "2nd level tiling", SecondLevelTileSizes,
                    SecondLevelDefaultTileSize);
    SecondLevelTileOpts++;
  }

  if (RegisterTiling) {
    Node =
        applyRegisterTiling(Node, RegisterTileSizes, RegisterDefaultTileSize);
    RegisterTileOpts++;
  }

  if (PollyVectorizerChoice == VECTORIZER_NONE)
    return Node;

  auto Space = isl::manage(isl_schedule_node_band_get_space(Node.get()));
  auto Dims = Space.dim(isl::dim::set);

  for (int i = Dims - 1; i >= 0; i--)
    if (Node.band_member_get_coincident(i)) {
      Node = prevectSchedBand(Node, i, PrevectorWidth);
      break;
    }

  return Node;
}

/// Permute the two dimensions of the isl map.
///
/// Permute @p DstPos and @p SrcPos dimensions of the isl map @p Map that
/// have type @p DimType.
///
/// @param Map     The isl map to be modified.
/// @param DimType The type of the dimensions.
/// @param DstPos  The first dimension.
/// @param SrcPos  The second dimension.
/// @return        The modified map.
isl::map permuteDimensions(isl::map Map, isl::dim DimType, unsigned DstPos,
                           unsigned SrcPos) {
  assert(DstPos < Map.dim(DimType) && SrcPos < Map.dim(DimType));
  if (DstPos == SrcPos)
    return Map;
  isl::id DimId;
  if (Map.has_tuple_id(DimType))
    DimId = Map.get_tuple_id(DimType);
  auto FreeDim = DimType == isl::dim::in ? isl::dim::out : isl::dim::in;
  isl::id FreeDimId;
  if (Map.has_tuple_id(FreeDim))
    FreeDimId = Map.get_tuple_id(FreeDim);
  auto MaxDim = std::max(DstPos, SrcPos);
  auto MinDim = std::min(DstPos, SrcPos);
  Map = Map.move_dims(FreeDim, 0, DimType, MaxDim, 1);
  Map = Map.move_dims(FreeDim, 0, DimType, MinDim, 1);
  Map = Map.move_dims(DimType, MinDim, FreeDim, 1, 1);
  Map = Map.move_dims(DimType, MaxDim, FreeDim, 0, 1);
  if (DimId)
    Map = Map.set_tuple_id(DimType, DimId);
  if (FreeDimId)
    Map = Map.set_tuple_id(FreeDim, FreeDimId);
  return Map;
}

/// Check the form of the access relation.
///
/// Check that the access relation @p AccMap has the form M[i][j], where i
/// is a @p FirstPos and j is a @p SecondPos.
///
/// @param AccMap    The access relation to be checked.
/// @param FirstPos  The index of the input dimension that is mapped to
///                  the first output dimension.
/// @param SecondPos The index of the input dimension that is mapped to the
///                  second output dimension.
/// @return          True in case @p AccMap has the expected form and false,
///                  otherwise.
static bool isMatMulOperandAcc(isl::set Domain, isl::map AccMap, int &FirstPos,
                               int &SecondPos) {
  isl::space Space = AccMap.get_space();
  isl::map Universe = isl::map::universe(Space);

  if (Space.dim(isl::dim::out) != 2)
    return false;

  // MatMul has the form:
  // for (i = 0; i < N; i++)
  //   for (j = 0; j < M; j++)
  //     for (k = 0; k < P; k++)
  //       C[i, j] += A[i, k] * B[k, j]
  //
  // Permutation of three outer loops: 3! = 6 possibilities.
  int FirstDims[] = {0, 0, 1, 1, 2, 2};
  int SecondDims[] = {1, 2, 2, 0, 0, 1};
  for (int i = 0; i < 6; i += 1) {
    auto PossibleMatMul =
        Universe.equate(isl::dim::in, FirstDims[i], isl::dim::out, 0)
            .equate(isl::dim::in, SecondDims[i], isl::dim::out, 1);

    AccMap = AccMap.intersect_domain(Domain);
    PossibleMatMul = PossibleMatMul.intersect_domain(Domain);

    // If AccMap spans entire domain (Non-partial write),
    // compute FirstPos and SecondPos.
    // If AccMap != PossibleMatMul here (the two maps have been gisted at
    // this point), it means that the writes are not complete, or in other
    // words, it is a Partial write and Partial writes must be rejected.
    if (AccMap.is_equal(PossibleMatMul)) {
      if (FirstPos != -1 && FirstPos != FirstDims[i])
        continue;
      FirstPos = FirstDims[i];
      if (SecondPos != -1 && SecondPos != SecondDims[i])
        continue;
      SecondPos = SecondDims[i];
      return true;
    }
  }

  return false;
}

/// Does the memory access represent a non-scalar operand of the matrix
/// multiplication.
///
/// Check that the memory access @p MemAccess is the read access to a non-scalar
/// operand of the matrix multiplication or its result.
///
/// @param MemAccess The memory access to be checked.
/// @param MMI       Parameters of the matrix multiplication operands.
/// @return          True in case the memory access represents the read access
///                  to a non-scalar operand of the matrix multiplication and
///                  false, otherwise.
static bool isMatMulNonScalarReadAccess(MemoryAccess *MemAccess,
                                        MatMulInfoTy &MMI) {
  if (!MemAccess->isLatestArrayKind() || !MemAccess->isRead())
    return false;
  auto AccMap = MemAccess->getLatestAccessRelation();
  isl::set StmtDomain = MemAccess->getStatement()->getDomain();
  if (isMatMulOperandAcc(StmtDomain, AccMap, MMI.i, MMI.j) && !MMI.ReadFromC) {
    MMI.ReadFromC = MemAccess;
    return true;
  }
  if (isMatMulOperandAcc(StmtDomain, AccMap, MMI.i, MMI.k) && !MMI.A) {
    MMI.A = MemAccess;
    return true;
  }
  if (isMatMulOperandAcc(StmtDomain, AccMap, MMI.k, MMI.j) && !MMI.B) {
    MMI.B = MemAccess;
    return true;
  }
  return false;
}

/// Check accesses to operands of the matrix multiplication.
///
/// Check that accesses of the SCoP statement, which corresponds to
/// the partial schedule @p PartialSchedule, are scalar in terms of loops
/// containing the matrix multiplication, in case they do not represent
/// accesses to the non-scalar operands of the matrix multiplication or
/// its result.
///
/// @param  PartialSchedule The partial schedule of the SCoP statement.
/// @param  MMI             Parameters of the matrix multiplication operands.
/// @return                 True in case the corresponding SCoP statement
///                         represents matrix multiplication and false,
///                         otherwise.
static bool containsOnlyMatrMultAcc(isl::map PartialSchedule,
                                    MatMulInfoTy &MMI) {
  auto InputDimId = PartialSchedule.get_tuple_id(isl::dim::in);
  auto *Stmt = static_cast<ScopStmt *>(InputDimId.get_user());
  unsigned OutDimNum = PartialSchedule.dim(isl::dim::out);
  assert(OutDimNum > 2 && "In case of the matrix multiplication the loop nest "
                          "and, consequently, the corresponding scheduling "
                          "functions have at least three dimensions.");
  auto MapI =
      permuteDimensions(PartialSchedule, isl::dim::out, MMI.i, OutDimNum - 1);
  auto MapJ =
      permuteDimensions(PartialSchedule, isl::dim::out, MMI.j, OutDimNum - 1);
  auto MapK =
      permuteDimensions(PartialSchedule, isl::dim::out, MMI.k, OutDimNum - 1);

  auto Accesses = getAccessesInOrder(*Stmt);
  for (auto *MemA = Accesses.begin(); MemA != Accesses.end() - 1; MemA++) {
    auto *MemAccessPtr = *MemA;
    if (MemAccessPtr->isLatestArrayKind() && MemAccessPtr != MMI.WriteToC &&
        !isMatMulNonScalarReadAccess(MemAccessPtr, MMI) &&
        !(MemAccessPtr->isStrideZero(MapI)) &&
        MemAccessPtr->isStrideZero(MapJ) && MemAccessPtr->isStrideZero(MapK))
      return false;
  }
  return true;
}

/// Check for dependencies corresponding to the matrix multiplication.
///
/// Check that there is only true dependence of the form
/// S(..., k, ...) -> S(..., k + 1, …), where S is the SCoP statement
/// represented by @p Schedule and k is @p Pos. Such a dependence corresponds
/// to the dependency produced by the matrix multiplication.
///
/// @param  Schedule The schedule of the SCoP statement.
/// @param  D The SCoP dependencies.
/// @param  Pos The parameter to describe an acceptable true dependence.
///             In case it has a negative value, try to determine its
///             acceptable value.
/// @return True in case dependencies correspond to the matrix multiplication
///         and false, otherwise.
static bool containsOnlyMatMulDep(isl::map Schedule, const Dependences *D,
                                  int &Pos) {
  isl::union_map Dep = D->getDependences(Dependences::TYPE_RAW);
  isl::union_map Red = D->getDependences(Dependences::TYPE_RED);
  if (Red)
    Dep = Dep.unite(Red);
  auto DomainSpace = Schedule.get_space().domain();
  auto Space = DomainSpace.map_from_domain_and_range(DomainSpace);
  auto Deltas = Dep.extract_map(Space).deltas();
  int DeltasDimNum = Deltas.dim(isl::dim::set);
  for (int i = 0; i < DeltasDimNum; i++) {
    auto Val = Deltas.plain_get_val_if_fixed(isl::dim::set, i);
    Pos = Pos < 0 && Val.is_one() ? i : Pos;
    if (Val.is_nan() || !(Val.is_zero() || (i == Pos && Val.is_one())))
      return false;
  }
  if (DeltasDimNum == 0 || Pos < 0)
    return false;
  return true;
}

/// Check if the SCoP statement could probably be optimized with analytical
/// modeling.
///
/// containsMatrMult tries to determine whether the following conditions
/// are true:
/// 1. The last memory access modeling an array, MA1, represents writing to
///    memory and has the form S(..., i1, ..., i2, ...) -> M(i1, i2) or
///    S(..., i2, ..., i1, ...) -> M(i1, i2), where S is the SCoP statement
///    under consideration.
/// 2. There is only one loop-carried true dependency, and it has the
///    form S(..., i3, ...) -> S(..., i3 + 1, ...), and there are no
///    loop-carried or anti dependencies.
/// 3. SCoP contains three access relations, MA2, MA3, and MA4 that represent
///    reading from memory and have the form S(..., i3, ...) -> M(i1, i3),
///    S(..., i3, ...) -> M(i3, i2), S(...) -> M(i1, i2), respectively,
///    and all memory accesses of the SCoP that are different from MA1, MA2,
///    MA3, and MA4 have stride 0, if the innermost loop is exchanged with any
///    of loops i1, i2 and i3.
///
/// @param PartialSchedule The PartialSchedule that contains a SCoP statement
///        to check.
/// @D     The SCoP dependencies.
/// @MMI   Parameters of the matrix multiplication operands.
static bool containsMatrMult(isl::map PartialSchedule, const Dependences *D,
                             MatMulInfoTy &MMI) {
  auto InputDimsId = PartialSchedule.get_tuple_id(isl::dim::in);
  auto *Stmt = static_cast<ScopStmt *>(InputDimsId.get_user());
  if (Stmt->size() <= 1)
    return false;

  auto Accesses = getAccessesInOrder(*Stmt);
  for (auto *MemA = Accesses.end() - 1; MemA != Accesses.begin(); MemA--) {
    auto *MemAccessPtr = *MemA;
    if (!MemAccessPtr->isLatestArrayKind())
      continue;
    if (!MemAccessPtr->isWrite())
      return false;
    auto AccMap = MemAccessPtr->getLatestAccessRelation();
    if (!isMatMulOperandAcc(Stmt->getDomain(), AccMap, MMI.i, MMI.j))
      return false;
    MMI.WriteToC = MemAccessPtr;
    break;
  }

  if (!containsOnlyMatMulDep(PartialSchedule, D, MMI.k))
    return false;

  if (!MMI.WriteToC || !containsOnlyMatrMultAcc(PartialSchedule, MMI))
    return false;

  if (!MMI.A || !MMI.B || !MMI.ReadFromC)
    return false;
  return true;
}

/// Permute two dimensions of the band node.
///
/// Permute FirstDim and SecondDim dimensions of the Node.
///
/// @param Node The band node to be modified.
/// @param FirstDim The first dimension to be permuted.
/// @param SecondDim The second dimension to be permuted.
static isl::schedule_node permuteBandNodeDimensions(isl::schedule_node Node,
                                                    unsigned FirstDim,
                                                    unsigned SecondDim) {
  assert(isl_schedule_node_get_type(Node.get()) == isl_schedule_node_band &&
         isl_schedule_node_band_n_member(Node.get()) >
             std::max(FirstDim, SecondDim));
  auto PartialSchedule =
      isl::manage(isl_schedule_node_band_get_partial_schedule(Node.get()));
  auto PartialScheduleFirstDim = PartialSchedule.get_union_pw_aff(FirstDim);
  auto PartialScheduleSecondDim = PartialSchedule.get_union_pw_aff(SecondDim);
  PartialSchedule =
      PartialSchedule.set_union_pw_aff(SecondDim, PartialScheduleFirstDim);
  PartialSchedule =
      PartialSchedule.set_union_pw_aff(FirstDim, PartialScheduleSecondDim);
  Node = isl::manage(isl_schedule_node_delete(Node.release()));
  return Node.insert_partial_schedule(PartialSchedule);
}

isl::schedule_node ScheduleTreeOptimizer::createMicroKernel(
    isl::schedule_node Node, MicroKernelParamsTy MicroKernelParams) {
  Node = applyRegisterTiling(Node, {MicroKernelParams.Mr, MicroKernelParams.Nr},
                             1);
  Node = Node.parent().parent();
  return permuteBandNodeDimensions(Node, 0, 1).child(0).child(0);
}

isl::schedule_node ScheduleTreeOptimizer::createMacroKernel(
    isl::schedule_node Node, MacroKernelParamsTy MacroKernelParams) {
  assert(isl_schedule_node_get_type(Node.get()) == isl_schedule_node_band);
  if (MacroKernelParams.Mc == 1 && MacroKernelParams.Nc == 1 &&
      MacroKernelParams.Kc == 1)
    return Node;
  int DimOutNum = isl_schedule_node_band_n_member(Node.get());
  std::vector<int> TileSizes(DimOutNum, 1);
  TileSizes[DimOutNum - 3] = MacroKernelParams.Mc;
  TileSizes[DimOutNum - 2] = MacroKernelParams.Nc;
  TileSizes[DimOutNum - 1] = MacroKernelParams.Kc;
  Node = tileNode(Node, "1st level tiling", TileSizes, 1);
  Node = Node.parent().parent();
  Node = permuteBandNodeDimensions(Node, DimOutNum - 2, DimOutNum - 1);
  Node = permuteBandNodeDimensions(Node, DimOutNum - 3, DimOutNum - 1);
  return Node.child(0).child(0);
}

/// Get the size of the widest type of the matrix multiplication operands
/// in bytes, including alignment padding.
///
/// @param MMI Parameters of the matrix multiplication operands.
/// @return The size of the widest type of the matrix multiplication operands
///         in bytes, including alignment padding.
static uint64_t getMatMulAlignTypeSize(MatMulInfoTy MMI) {
  auto *S = MMI.A->getStatement()->getParent();
  auto &DL = S->getFunction().getParent()->getDataLayout();
  auto ElementSizeA = DL.getTypeAllocSize(MMI.A->getElementType());
  auto ElementSizeB = DL.getTypeAllocSize(MMI.B->getElementType());
  auto ElementSizeC = DL.getTypeAllocSize(MMI.WriteToC->getElementType());
  return std::max({ElementSizeA, ElementSizeB, ElementSizeC});
}

/// Get the size of the widest type of the matrix multiplication operands
/// in bits.
///
/// @param MMI Parameters of the matrix multiplication operands.
/// @return The size of the widest type of the matrix multiplication operands
///         in bits.
static uint64_t getMatMulTypeSize(MatMulInfoTy MMI) {
  auto *S = MMI.A->getStatement()->getParent();
  auto &DL = S->getFunction().getParent()->getDataLayout();
  auto ElementSizeA = DL.getTypeSizeInBits(MMI.A->getElementType());
  auto ElementSizeB = DL.getTypeSizeInBits(MMI.B->getElementType());
  auto ElementSizeC = DL.getTypeSizeInBits(MMI.WriteToC->getElementType());
  return std::max({ElementSizeA, ElementSizeB, ElementSizeC});
}

/// Get parameters of the BLIS micro kernel.
///
/// We choose the Mr and Nr parameters of the micro kernel to be large enough
/// such that no stalls caused by the combination of latencies and dependencies
/// are introduced during the updates of the resulting matrix of the matrix
/// multiplication. However, they should also be as small as possible to
/// release more registers for entries of multiplied matrices.
///
/// @param TTI Target Transform Info.
/// @param MMI Parameters of the matrix multiplication operands.
/// @return The structure of type MicroKernelParamsTy.
/// @see MicroKernelParamsTy
static struct MicroKernelParamsTy
getMicroKernelParams(const TargetTransformInfo *TTI, MatMulInfoTy MMI) {
  assert(TTI && "The target transform info should be provided.");

  // Nvec - Number of double-precision floating-point numbers that can be hold
  // by a vector register. Use 2 by default.
  long RegisterBitwidth = VectorRegisterBitwidth;

  if (RegisterBitwidth == -1)
    RegisterBitwidth = TTI->getRegisterBitWidth(true);
  auto ElementSize = getMatMulTypeSize(MMI);
  assert(ElementSize > 0 && "The element size of the matrix multiplication "
                            "operands should be greater than zero.");
  auto Nvec = RegisterBitwidth / ElementSize;
  if (Nvec == 0)
    Nvec = 2;
  int Nr =
      ceil(sqrt(Nvec * LatencyVectorFma * ThroughputVectorFma) / Nvec) * Nvec;
  int Mr = ceil(Nvec * LatencyVectorFma * ThroughputVectorFma / Nr);
  return {Mr, Nr};
}

namespace {
/// Determine parameters of the target cache.
///
/// @param TTI Target Transform Info.
void getTargetCacheParameters(const llvm::TargetTransformInfo *TTI) {
  auto L1DCache = llvm::TargetTransformInfo::CacheLevel::L1D;
  auto L2DCache = llvm::TargetTransformInfo::CacheLevel::L2D;
  if (FirstCacheLevelSize == -1) {
    if (TTI->getCacheSize(L1DCache).hasValue())
      FirstCacheLevelSize = TTI->getCacheSize(L1DCache).getValue();
    else
      FirstCacheLevelSize = static_cast<int>(FirstCacheLevelDefaultSize);
  }
  if (SecondCacheLevelSize == -1) {
    if (TTI->getCacheSize(L2DCache).hasValue())
      SecondCacheLevelSize = TTI->getCacheSize(L2DCache).getValue();
    else
      SecondCacheLevelSize = static_cast<int>(SecondCacheLevelDefaultSize);
  }
  if (FirstCacheLevelAssociativity == -1) {
    if (TTI->getCacheAssociativity(L1DCache).hasValue())
      FirstCacheLevelAssociativity =
          TTI->getCacheAssociativity(L1DCache).getValue();
    else
      FirstCacheLevelAssociativity =
          static_cast<int>(FirstCacheLevelDefaultAssociativity);
  }
  if (SecondCacheLevelAssociativity == -1) {
    if (TTI->getCacheAssociativity(L2DCache).hasValue())
      SecondCacheLevelAssociativity =
          TTI->getCacheAssociativity(L2DCache).getValue();
    else
      SecondCacheLevelAssociativity =
          static_cast<int>(SecondCacheLevelDefaultAssociativity);
  }
}
} // namespace

/// Get parameters of the BLIS macro kernel.
///
/// During the computation of matrix multiplication, blocks of partitioned
/// matrices are mapped to different layers of the memory hierarchy.
/// To optimize data reuse, blocks should be ideally kept in cache between
/// iterations. Since parameters of the macro kernel determine sizes of these
/// blocks, there are upper and lower bounds on these parameters.
///
/// @param TTI Target Transform Info.
/// @param MicroKernelParams Parameters of the micro-kernel
///                          to be taken into account.
/// @param MMI Parameters of the matrix multiplication operands.
/// @return The structure of type MacroKernelParamsTy.
/// @see MacroKernelParamsTy
/// @see MicroKernelParamsTy
static struct MacroKernelParamsTy
getMacroKernelParams(const llvm::TargetTransformInfo *TTI,
                     const MicroKernelParamsTy &MicroKernelParams,
                     MatMulInfoTy MMI) {
  getTargetCacheParameters(TTI);
  // According to www.cs.utexas.edu/users/flame/pubs/TOMS-BLIS-Analytical.pdf,
  // it requires information about the first two levels of a cache to determine
  // all the parameters of a macro-kernel. It also checks that an associativity
  // degree of a cache level is greater than two. Otherwise, another algorithm
  // for determination of the parameters should be used.
  if (!(MicroKernelParams.Mr > 0 && MicroKernelParams.Nr > 0 &&
        FirstCacheLevelSize > 0 && SecondCacheLevelSize > 0 &&
        FirstCacheLevelAssociativity > 2 && SecondCacheLevelAssociativity > 2))
    return {1, 1, 1};
  // The quotient should be greater than zero.
  if (PollyPatternMatchingNcQuotient <= 0)
    return {1, 1, 1};
  int Car = floor(
      (FirstCacheLevelAssociativity - 1) /
      (1 + static_cast<double>(MicroKernelParams.Nr) / MicroKernelParams.Mr));

  // Car can be computed to be zero since it is floor to int.
  // On Mac OS, division by 0 does not raise a signal. This causes negative
  // tile sizes to be computed. Prevent division by Cac==0 by early returning
  // if this happens.
  if (Car == 0)
    return {1, 1, 1};

  auto ElementSize = getMatMulAlignTypeSize(MMI);
  assert(ElementSize > 0 && "The element size of the matrix multiplication "
                            "operands should be greater than zero.");
  int Kc = (Car * FirstCacheLevelSize) /
           (MicroKernelParams.Mr * FirstCacheLevelAssociativity * ElementSize);
  double Cac =
      static_cast<double>(Kc * ElementSize * SecondCacheLevelAssociativity) /
      SecondCacheLevelSize;
  int Mc = floor((SecondCacheLevelAssociativity - 2) / Cac);
  int Nc = PollyPatternMatchingNcQuotient * MicroKernelParams.Nr;

  assert(Mc > 0 && Nc > 0 && Kc > 0 &&
         "Matrix block sizes should be  greater than zero");
  return {Mc, Nc, Kc};
}

/// Create an access relation that is specific to
///        the matrix multiplication pattern.
///
/// Create an access relation of the following form:
/// [O0, O1, O2, O3, O4, O5, O6, O7, O8] -> [OI, O5, OJ]
/// where I is @p FirstDim, J is @p SecondDim.
///
/// It can be used, for example, to create relations that helps to consequently
/// access elements of operands of a matrix multiplication after creation of
/// the BLIS micro and macro kernels.
///
/// @see ScheduleTreeOptimizer::createMicroKernel
/// @see ScheduleTreeOptimizer::createMacroKernel
///
/// Subsequently, the described access relation is applied to the range of
/// @p MapOldIndVar, that is used to map original induction variables to
/// the ones, which are produced by schedule transformations. It helps to
/// define relations using a new space and, at the same time, keep them
/// in the original one.
///
/// @param MapOldIndVar The relation, which maps original induction variables
///                     to the ones, which are produced by schedule
///                     transformations.
/// @param FirstDim, SecondDim The input dimensions that are used to define
///        the specified access relation.
/// @return The specified access relation.
isl::map getMatMulAccRel(isl::map MapOldIndVar, unsigned FirstDim,
                         unsigned SecondDim) {
  auto AccessRelSpace = isl::space(MapOldIndVar.get_ctx(), 0, 9, 3);
  auto AccessRel = isl::map::universe(AccessRelSpace);
  AccessRel = AccessRel.equate(isl::dim::in, FirstDim, isl::dim::out, 0);
  AccessRel = AccessRel.equate(isl::dim::in, 5, isl::dim::out, 1);
  AccessRel = AccessRel.equate(isl::dim::in, SecondDim, isl::dim::out, 2);
  return MapOldIndVar.apply_range(AccessRel);
}

isl::schedule_node createExtensionNode(isl::schedule_node Node,
                                       isl::map ExtensionMap) {
  auto Extension = isl::union_map(ExtensionMap);
  auto NewNode = isl::schedule_node::from_extension(Extension);
  return Node.graft_before(NewNode);
}

/// Apply the packing transformation.
///
/// The packing transformation can be described as a data-layout
/// transformation that requires to introduce a new array, copy data
/// to the array, and change memory access locations to reference the array.
/// It can be used to ensure that elements of the new array are read in-stride
/// access, aligned to cache lines boundaries, and preloaded into certain cache
/// levels.
///
/// As an example let us consider the packing of the array A that would help
/// to read its elements with in-stride access. An access to the array A
/// is represented by an access relation that has the form
/// S[i, j, k] -> A[i, k]. The scheduling function of the SCoP statement S has
/// the form S[i,j, k] -> [floor((j mod Nc) / Nr), floor((i mod Mc) / Mr),
/// k mod Kc, j mod Nr, i mod Mr].
///
/// To ensure that elements of the array A are read in-stride access, we add
/// a new array Packed_A[Mc/Mr][Kc][Mr] to the SCoP, using
/// Scop::createScopArrayInfo, change the access relation
/// S[i, j, k] -> A[i, k] to
/// S[i, j, k] -> Packed_A[floor((i mod Mc) / Mr), k mod Kc, i mod Mr], using
/// MemoryAccess::setNewAccessRelation, and copy the data to the array, using
/// the copy statement created by Scop::addScopStmt.
///
/// @param Node The schedule node to be optimized.
/// @param MapOldIndVar The relation, which maps original induction variables
///                     to the ones, which are produced by schedule
///                     transformations.
/// @param MicroParams, MacroParams Parameters of the BLIS kernel
///                                 to be taken into account.
/// @param MMI Parameters of the matrix multiplication operands.
/// @return The optimized schedule node.
static isl::schedule_node
optimizeDataLayoutMatrMulPattern(isl::schedule_node Node, isl::map MapOldIndVar,
                                 MicroKernelParamsTy MicroParams,
                                 MacroKernelParamsTy MacroParams,
                                 MatMulInfoTy &MMI) {
  auto InputDimsId = MapOldIndVar.get_tuple_id(isl::dim::in);
  auto *Stmt = static_cast<ScopStmt *>(InputDimsId.get_user());

  // Create a copy statement that corresponds to the memory access to the
  // matrix B, the second operand of the matrix multiplication.
  Node = Node.parent().parent().parent().parent().parent().parent();
  Node = isl::manage(isl_schedule_node_band_split(Node.release(), 2)).child(0);
  auto AccRel = getMatMulAccRel(MapOldIndVar, 3, 7);
  unsigned FirstDimSize = MacroParams.Nc / MicroParams.Nr;
  unsigned SecondDimSize = MacroParams.Kc;
  unsigned ThirdDimSize = MicroParams.Nr;
  auto *SAI = Stmt->getParent()->createScopArrayInfo(
      MMI.B->getElementType(), "Packed_B",
      {FirstDimSize, SecondDimSize, ThirdDimSize});
  AccRel = AccRel.set_tuple_id(isl::dim::out, SAI->getBasePtrId());
  auto OldAcc = MMI.B->getLatestAccessRelation();
  MMI.B->setNewAccessRelation(AccRel);
  auto ExtMap = MapOldIndVar.project_out(isl::dim::out, 2,
                                         MapOldIndVar.dim(isl::dim::out) - 2);
  ExtMap = ExtMap.reverse();
  ExtMap = ExtMap.fix_si(isl::dim::out, MMI.i, 0);
  auto Domain = Stmt->getDomain();

  // Restrict the domains of the copy statements to only execute when also its
  // originating statement is executed.
  auto DomainId = Domain.get_tuple_id();
  auto *NewStmt = Stmt->getParent()->addScopStmt(
      OldAcc, MMI.B->getLatestAccessRelation(), Domain);
  ExtMap = ExtMap.set_tuple_id(isl::dim::out, DomainId);
  ExtMap = ExtMap.intersect_range(Domain);
  ExtMap = ExtMap.set_tuple_id(isl::dim::out, NewStmt->getDomainId());
  Node = createExtensionNode(Node, ExtMap);

  // Create a copy statement that corresponds to the memory access
  // to the matrix A, the first operand of the matrix multiplication.
  Node = Node.child(0);
  AccRel = getMatMulAccRel(MapOldIndVar, 4, 6);
  FirstDimSize = MacroParams.Mc / MicroParams.Mr;
  ThirdDimSize = MicroParams.Mr;
  SAI = Stmt->getParent()->createScopArrayInfo(
      MMI.A->getElementType(), "Packed_A",
      {FirstDimSize, SecondDimSize, ThirdDimSize});
  AccRel = AccRel.set_tuple_id(isl::dim::out, SAI->getBasePtrId());
  OldAcc = MMI.A->getLatestAccessRelation();
  MMI.A->setNewAccessRelation(AccRel);
  ExtMap = MapOldIndVar.project_out(isl::dim::out, 3,
                                    MapOldIndVar.dim(isl::dim::out) - 3);
  ExtMap = ExtMap.reverse();
  ExtMap = ExtMap.fix_si(isl::dim::out, MMI.j, 0);
  NewStmt = Stmt->getParent()->addScopStmt(
      OldAcc, MMI.A->getLatestAccessRelation(), Domain);

  // Restrict the domains of the copy statements to only execute when also its
  // originating statement is executed.
  ExtMap = ExtMap.set_tuple_id(isl::dim::out, DomainId);
  ExtMap = ExtMap.intersect_range(Domain);
  ExtMap = ExtMap.set_tuple_id(isl::dim::out, NewStmt->getDomainId());
  Node = createExtensionNode(Node, ExtMap);
  return Node.child(0).child(0).child(0).child(0).child(0);
}

/// Get a relation mapping induction variables produced by schedule
/// transformations to the original ones.
///
/// @param Node The schedule node produced as the result of creation
///        of the BLIS kernels.
/// @param MicroKernelParams, MacroKernelParams Parameters of the BLIS kernel
///                                             to be taken into account.
/// @return  The relation mapping original induction variables to the ones
///          produced by schedule transformation.
/// @see ScheduleTreeOptimizer::createMicroKernel
/// @see ScheduleTreeOptimizer::createMacroKernel
/// @see getMacroKernelParams
isl::map
getInductionVariablesSubstitution(isl::schedule_node Node,
                                  MicroKernelParamsTy MicroKernelParams,
                                  MacroKernelParamsTy MacroKernelParams) {
  auto Child = Node.child(0);
  auto UnMapOldIndVar = Child.get_prefix_schedule_union_map();
  auto MapOldIndVar = isl::map::from_union_map(UnMapOldIndVar);
  if (MapOldIndVar.dim(isl::dim::out) > 9)
    return MapOldIndVar.project_out(isl::dim::out, 0,
                                    MapOldIndVar.dim(isl::dim::out) - 9);
  return MapOldIndVar;
}

/// Isolate a set of partial tile prefixes and unroll the isolated part.
///
/// The set should ensure that it contains only partial tile prefixes that have
/// exactly Mr x Nr iterations of the two innermost loops produced by
/// the optimization of the matrix multiplication. Mr and Nr are parameters of
/// the micro-kernel.
///
/// In case of parametric bounds, this helps to auto-vectorize the unrolled
/// innermost loops, using the SLP vectorizer.
///
/// @param Node              The schedule node to be modified.
/// @param MicroKernelParams Parameters of the micro-kernel
///                          to be taken into account.
/// @return The modified isl_schedule_node.
static isl::schedule_node
isolateAndUnrollMatMulInnerLoops(isl::schedule_node Node,
                                 struct MicroKernelParamsTy MicroKernelParams) {
  isl::schedule_node Child = Node.get_child(0);
  isl::union_map UnMapOldIndVar = Child.get_prefix_schedule_relation();
  isl::set Prefix = isl::map::from_union_map(UnMapOldIndVar).range();
  unsigned Dims = Prefix.dim(isl::dim::set);
  Prefix = Prefix.project_out(isl::dim::set, Dims - 1, 1);
  Prefix = getPartialTilePrefixes(Prefix, MicroKernelParams.Nr);
  Prefix = getPartialTilePrefixes(Prefix, MicroKernelParams.Mr);

  isl::union_set IsolateOption =
      getIsolateOptions(Prefix.add_dims(isl::dim::set, 3), 3);
  isl::ctx Ctx = Node.get_ctx();
  auto Options = IsolateOption.unite(getDimOptions(Ctx, "unroll"));
  Options = Options.unite(getUnrollIsolatedSetOptions(Ctx));
  Node = Node.band_set_ast_build_options(Options);
  Node = Node.parent().parent().parent();
  IsolateOption = getIsolateOptions(Prefix, 3);
  Options = IsolateOption.unite(getDimOptions(Ctx, "separate"));
  Node = Node.band_set_ast_build_options(Options);
  Node = Node.child(0).child(0).child(0);
  return Node;
}

/// Mark @p BasePtr with "Inter iteration alias-free" mark node.
///
/// @param Node The child of the mark node to be inserted.
/// @param BasePtr The pointer to be marked.
/// @return The modified isl_schedule_node.
static isl::schedule_node markInterIterationAliasFree(isl::schedule_node Node,
                                                      Value *BasePtr) {
  if (!BasePtr)
    return Node;

  auto Id =
      isl::id::alloc(Node.get_ctx(), "Inter iteration alias-free", BasePtr);
  return Node.insert_mark(Id).child(0);
}

/// Insert "Loop Vectorizer Disabled" mark node.
///
/// @param Node The child of the mark node to be inserted.
/// @return The modified isl_schedule_node.
static isl::schedule_node markLoopVectorizerDisabled(isl::schedule_node Node) {
  auto Id = isl::id::alloc(Node.get_ctx(), "Loop Vectorizer Disabled", nullptr);
  return Node.insert_mark(Id).child(0);
}

/// Restore the initial ordering of dimensions of the band node
///
/// In case the band node represents all the dimensions of the iteration
/// domain, recreate the band node to restore the initial ordering of the
/// dimensions.
///
/// @param Node The band node to be modified.
/// @return The modified schedule node.
static isl::schedule_node
getBandNodeWithOriginDimOrder(isl::schedule_node Node) {
  assert(isl_schedule_node_get_type(Node.get()) == isl_schedule_node_band);
  if (isl_schedule_node_get_type(Node.child(0).get()) != isl_schedule_node_leaf)
    return Node;
  auto Domain = Node.get_universe_domain();
  assert(isl_union_set_n_set(Domain.get()) == 1);
  if (Node.get_schedule_depth() != 0 ||
      (isl::set(Domain).dim(isl::dim::set) !=
       isl_schedule_node_band_n_member(Node.get())))
    return Node;
  Node = isl::manage(isl_schedule_node_delete(Node.copy()));
  auto PartialSchedulePwAff = Domain.identity_union_pw_multi_aff();
  auto PartialScheduleMultiPwAff =
      isl::multi_union_pw_aff(PartialSchedulePwAff);
  PartialScheduleMultiPwAff =
      PartialScheduleMultiPwAff.reset_tuple_id(isl::dim::set);
  return Node.insert_partial_schedule(PartialScheduleMultiPwAff);
}

isl::schedule_node
ScheduleTreeOptimizer::optimizeMatMulPattern(isl::schedule_node Node,
                                             const TargetTransformInfo *TTI,
                                             MatMulInfoTy &MMI) {
  assert(TTI && "The target transform info should be provided.");
  Node = markInterIterationAliasFree(
      Node, MMI.WriteToC->getLatestScopArrayInfo()->getBasePtr());
  int DimOutNum = isl_schedule_node_band_n_member(Node.get());
  assert(DimOutNum > 2 && "In case of the matrix multiplication the loop nest "
                          "and, consequently, the corresponding scheduling "
                          "functions have at least three dimensions.");
  Node = getBandNodeWithOriginDimOrder(Node);
  Node = permuteBandNodeDimensions(Node, MMI.i, DimOutNum - 3);
  int NewJ = MMI.j == DimOutNum - 3 ? MMI.i : MMI.j;
  int NewK = MMI.k == DimOutNum - 3 ? MMI.i : MMI.k;
  Node = permuteBandNodeDimensions(Node, NewJ, DimOutNum - 2);
  NewK = NewK == DimOutNum - 2 ? NewJ : NewK;
  Node = permuteBandNodeDimensions(Node, NewK, DimOutNum - 1);
  auto MicroKernelParams = getMicroKernelParams(TTI, MMI);
  auto MacroKernelParams = getMacroKernelParams(TTI, MicroKernelParams, MMI);
  Node = createMacroKernel(Node, MacroKernelParams);
  Node = createMicroKernel(Node, MicroKernelParams);
  if (MacroKernelParams.Mc == 1 || MacroKernelParams.Nc == 1 ||
      MacroKernelParams.Kc == 1)
    return Node;
  auto MapOldIndVar = getInductionVariablesSubstitution(Node, MicroKernelParams,
                                                        MacroKernelParams);
  if (!MapOldIndVar)
    return Node;
  Node = markLoopVectorizerDisabled(Node.parent()).child(0);
  Node = isolateAndUnrollMatMulInnerLoops(Node, MicroKernelParams);
  return optimizeDataLayoutMatrMulPattern(Node, MapOldIndVar, MicroKernelParams,
                                          MacroKernelParams, MMI);
}

bool ScheduleTreeOptimizer::isMatrMultPattern(isl::schedule_node Node,
                                              const Dependences *D,
                                              MatMulInfoTy &MMI) {
  auto PartialSchedule = isl::manage(
      isl_schedule_node_band_get_partial_schedule_union_map(Node.get()));
  Node = Node.child(0);
  auto LeafType = isl_schedule_node_get_type(Node.get());
  Node = Node.parent();
  if (LeafType != isl_schedule_node_leaf ||
      isl_schedule_node_band_n_member(Node.get()) < 3 ||
      Node.get_schedule_depth() != 0 ||
      isl_union_map_n_map(PartialSchedule.get()) != 1)
    return false;
  auto NewPartialSchedule = isl::map::from_union_map(PartialSchedule);
  if (containsMatrMult(NewPartialSchedule, D, MMI))
    return true;
  return false;
}

__isl_give isl_schedule_node *
ScheduleTreeOptimizer::optimizeBand(__isl_take isl_schedule_node *Node,
                                    void *User) {
  if (!isTileableBandNode(isl::manage_copy(Node)))
    return Node;

  const OptimizerAdditionalInfoTy *OAI =
      static_cast<const OptimizerAdditionalInfoTy *>(User);

  MatMulInfoTy MMI;
  if (PMBasedOpts && User &&
      isMatrMultPattern(isl::manage_copy(Node), OAI->D, MMI)) {
    LLVM_DEBUG(dbgs() << "The matrix multiplication pattern was detected\n");
    MatMulOpts++;
    return optimizeMatMulPattern(isl::manage(Node), OAI->TTI, MMI).release();
  }

  return standardBandOpts(isl::manage(Node), User).release();
}

isl::schedule
ScheduleTreeOptimizer::optimizeSchedule(isl::schedule Schedule,
                                        const OptimizerAdditionalInfoTy *OAI) {
  auto Root = Schedule.get_root();
  Root = optimizeScheduleNode(Root, OAI);
  return Root.get_schedule();
}

isl::schedule_node ScheduleTreeOptimizer::optimizeScheduleNode(
    isl::schedule_node Node, const OptimizerAdditionalInfoTy *OAI) {
  Node = isl::manage(isl_schedule_node_map_descendant_bottom_up(
      Node.release(), optimizeBand,
      const_cast<void *>(static_cast<const void *>(OAI))));
  return Node;
}

bool ScheduleTreeOptimizer::isProfitableSchedule(Scop &S,
                                                 isl::schedule NewSchedule) {
  // To understand if the schedule has been optimized we check if the schedule
  // has changed at all.
  // TODO: We can improve this by tracking if any necessarily beneficial
  // transformations have been performed. This can e.g. be tiling, loop
  // interchange, or ...) We can track this either at the place where the
  // transformation has been performed or, in case of automatic ILP based
  // optimizations, by comparing (yet to be defined) performance metrics
  // before/after the scheduling optimizer
  // (e.g., #stride-one accesses)
  if (S.containsExtensionNode(NewSchedule))
    return true;
  auto NewScheduleMap = NewSchedule.get_map();
  auto OldSchedule = S.getSchedule();
  assert(OldSchedule && "Only IslScheduleOptimizer can insert extension nodes "
                        "that make Scop::getSchedule() return nullptr.");
  bool changed = !OldSchedule.is_equal(NewScheduleMap);
  return changed;
}

namespace {

class IslScheduleOptimizer : public ScopPass {
public:
  static char ID;

  explicit IslScheduleOptimizer() : ScopPass(ID) {}

  ~IslScheduleOptimizer() override { isl_schedule_free(LastSchedule); }

  /// Optimize the schedule of the SCoP @p S.
  bool runOnScop(Scop &S) override;

  /// Print the new schedule for the SCoP @p S.
  void printScop(raw_ostream &OS, Scop &S) const override;

  /// Register all analyses and transformation required.
  void getAnalysisUsage(AnalysisUsage &AU) const override;

  /// Release the internal memory.
  void releaseMemory() override {
    isl_schedule_free(LastSchedule);
    LastSchedule = nullptr;
  }

private:
  isl_schedule *LastSchedule = nullptr;
};
} // namespace

char IslScheduleOptimizer::ID = 0;

/// Collect statistics for the schedule tree.
///
/// @param Schedule The schedule tree to analyze. If not a schedule tree it is
/// ignored.
/// @param Version  The version of the schedule tree that is analyzed.
///                 0 for the original schedule tree before any transformation.
///                 1 for the schedule tree after isl's rescheduling.
///                 2 for the schedule tree after optimizations are applied
///                 (tiling, pattern matching)
static void walkScheduleTreeForStatistics(isl::schedule Schedule, int Version) {
  auto Root = Schedule.get_root();
  if (!Root)
    return;

  isl_schedule_node_foreach_descendant_top_down(
      Root.get(),
      [](__isl_keep isl_schedule_node *nodeptr, void *user) -> isl_bool {
        isl::schedule_node Node = isl::manage_copy(nodeptr);
        int Version = *static_cast<int *>(user);

        switch (isl_schedule_node_get_type(Node.get())) {
        case isl_schedule_node_band: {
          NumBands[Version]++;
          if (isl_schedule_node_band_get_permutable(Node.get()) ==
              isl_bool_true)
            NumPermutable[Version]++;

          int CountMembers = isl_schedule_node_band_n_member(Node.get());
          NumBandMembers[Version] += CountMembers;
          for (int i = 0; i < CountMembers; i += 1) {
            if (Node.band_member_get_coincident(i))
              NumCoincident[Version]++;
          }
          break;
        }

        case isl_schedule_node_filter:
          NumFilters[Version]++;
          break;

        case isl_schedule_node_extension:
          NumExtension[Version]++;
          break;

        default:
          break;
        }

        return isl_bool_true;
      },
      &Version);
}

/// Walk the schedule tree and check if the tree
/// has been annotated by loop tactics.
///
/// @param schedule: New schedule obtained from loop tactics.
/// @param s: Annotation to look-up for.
bool lookUpScheduleTree(isl::schedule schedule, std::string s) {

  assert(schedule && "empty schedule!");
  isl::schedule_node root = schedule.get_root();

  struct payload {
    bool isOptimized = false;
    std::string name = "empty";
  } p;

  p.name = s;

  isl_schedule_node_foreach_descendant_top_down(
      root.get(),
      [](__isl_keep isl_schedule_node *nodePtr, void *user) -> isl_bool {
        payload *p = static_cast<payload *>(user);

        isl::schedule_node node = isl::manage_copy(nodePtr);
        if (isl_schedule_node_get_type(node.get()) == isl_schedule_node_mark) {
          isl::id id = node.mark_get_id();
          std::string idAsString = id.to_str();
          idAsString = idAsString.substr(0, idAsString.find("@"));
          if ((idAsString.compare(p->name) == 0)) {
            p->isOptimized = true;
          }
        }
        return isl_bool_true;
      },
      &p);

  return p.isOptimized;
}

/// utility function for wrapPatternDFSPreorder
///
/// @param ctx
/// @param marker: Marker to insert
/// @param node: Root of the subtree to inspect.
/// @param pattern: Pattern to look-up in the subtree.
/// @param runOnMatch: callback after structural matching.
template <typename T>
isl::schedule_node wrapPattern(isl::ctx ctx, Payload<T> *payload,
                               const std::string &marker,
                               isl::schedule_node node,
                               const matchers::ScheduleNodeMatcher &pattern,
                               std::function<bool(void)> runOnMatch) {

  if (matchers::ScheduleNodeMatcher::isMatching(pattern, node)) {
    if (runOnMatch && runOnMatch()) {
      node = node.insert_mark(isl::id::alloc(ctx, marker, payload));
    }
  }
  return node;
}

/// utility function for wrapPatternDFSPreorder
///
/// @param ctx
/// @param marker: Marker to insert
/// @param node: Root of the subtree to inspect.
/// @param pattern: Pattern to look-up in the subtree.
template <typename T>
isl::schedule_node wrapPattern(isl::ctx ctx, Payload<T> *payload,
                               const std::string &marker,
                               isl::schedule_node node,
                               const matchers::ScheduleNodeMatcher &pattern) {
  // LLVM_DEBUG(dbgs() << "node :" << node.to_str() << "\n");
  if (matchers::ScheduleNodeMatcher::isMatching(pattern, node)) {
    node = node.insert_mark(isl::id::alloc(ctx, marker, payload));
    LLVM_DEBUG(dbgs() << "MATCHED!!!\n");
  }
  return node;
}

template <typename T>
isl::schedule_node
wrapPatternDFSPreorder(isl::ctx ctx, Payload<T> *payload,
                       const std::string &marker, isl::schedule_node node,
                       const matchers::ScheduleNodeMatcher &pattern,
                       std::function<bool(void)> runOnMatch = nullptr) {
  if (runOnMatch)
    node = wrapPattern(ctx, payload, marker, node, pattern, runOnMatch);
  else
    node = wrapPattern(ctx, payload, marker, node, pattern);

  // if ((isl_schedule_node_get_type(node.get()) == isl_schedule_node_mark))
  //  return node;

  for (int i = 0; i < node.n_children(); i++) {
    node = wrapPatternDFSPreorder(ctx, payload, marker, node.child(i), pattern,
                                  runOnMatch)
               .parent();
  }
  return node;
}

/// utility function for replaceDFSPreorderRepeatedly
///
/// @param node: Current node where to start cutting.
/// @param replacement: Subtree to be attached after @p node.
///
/// NOTE: This is not always possible. Cutting children
/// in set or sequence is not allowed by ISL and as a consequence
/// by Loop Tactics.
isl::schedule_node rebuild(isl::schedule_node node,
                           const builders::ScheduleNodeBuilder &replacement) {

  node = node.cut();
  node = replacement.insertAt(node);
  return node;
}

/// utility function for replaceDFSPreorderRepeatedly.
///
/// @param node: Root of the subtree to inspect.
/// @param pattern: Pattern to look-up in the subtree.
/// @param replacement: Replacement to be applied in case
/// of a match with @p pattern.
isl::schedule_node
replaceRepeatedly(isl::schedule_node node,
                  const matchers::ScheduleNodeMatcher &pattern,
                  const builders::ScheduleNodeBuilder &replacement) {

  while (matchers::ScheduleNodeMatcher::isMatching(pattern, node)) {
    node = rebuild(node, replacement);
    // XXX: if we insert a single mark node, we end up in
    // an infinate loop, since they subtree under the mark will always
    // match the matcher. Escape this skipping the mark node and the
    // root node of the matcher.
    // if (isl_schedule_node_get_type(node.get()) == isl_schedule_node_mark)
    //  node = node.child(0).child(0);
  }
  return node;
}

/// walk the schedule tree starting from "node" and in
/// case of a match with the matcher "pattern" modify
/// the schedule tree using the builder "replacement".
///
/// @param node: Root of the subtree to inspect.
/// @param pattern: Pattern to look-up in the subtree.
/// @param replacement: Replacement to be applied in case of
/// a match with @p pattern.
isl::schedule_node
replaceDFSPreorderRepeatedly(isl::schedule_node node,
                             const matchers::ScheduleNodeMatcher &pattern,
                             const builders::ScheduleNodeBuilder &replacement) {

  node = replaceRepeatedly(node, pattern, replacement);
  for (int i = 0; i < node.n_children(); i++) {
    node = replaceDFSPreorderRepeatedly(node.child(i), pattern, replacement)
               .parent();
  }
  return node;
}

/// Helper function for replaceDFSPreorderOnce.
isl::schedule_node
replaceOnce(isl::schedule_node node,
            const matchers::ScheduleNodeMatcher &pattern,
            const builders::ScheduleNodeBuilder &replacement) {
  if (matchers::ScheduleNodeMatcher::isMatching(pattern, node)) {
    node = rebuild(node, replacement);
  }
  return node;
}

/// walk the schedule tree starting from "node" and in
/// case of a match with the matcher "pattern" modify
/// the schedule tree using the builder "replacement".
///
/// @param node: Root of the subtree to inspect.
/// @param pattern: Pattern to look-up in the subtree.
/// @param replacement: Replacement to be applied in case of
/// a match with @p pattern.
/// Apply the builder only once.
isl::schedule_node
replaceDFSPreorderOnce(isl::schedule_node node,
                       const matchers::ScheduleNodeMatcher &pattern,
                       const builders::ScheduleNodeBuilder &replacement) {
  node = replaceOnce(node, pattern, replacement);
  for (int i = 0; i < node.n_children(); ++i) {
    node = replaceDFSPreorderOnce(node.child(i), pattern, replacement).parent();
  }
  return node;
}

/// Get the memory access using tagged accesses.
MemoryAccess *getMemoryAccessFromTagged(const Scop &s, isl::map schedule,
                                        std::vector<isl::space> x,
                                        std::vector<isl::space> y,
                                        std::string type) {

  isl::space targetSpace;
  for (size_t i = 0; i < x.size(); i++)
    for (size_t j = 0; j < y.size(); j++)
      if (x[i].is_equal(y[j])) {
        targetSpace = x[i];
        break;
      }

  if (targetSpace.is_null()) {
    LLVM_DEBUG(dbgs() << "schedule :" << schedule.to_str() << "\n");
    LLVM_DEBUG(dbgs() << "x.size :" << x.size() << "\n");
    LLVM_DEBUG(dbgs() << "y.size :" << y.size() << "\n");
    LLVM_DEBUG(dbgs() << "type : " << type << "\n");
    assert(0 && "intersection failed!");
  }

  enum MemoryAccess::AccessType AccessTy;
  if (type.compare("r") == 0)
    AccessTy = MemoryAccess::AccessType::READ;
  else
    AccessTy = MemoryAccess::AccessType::MUST_WRITE;

  for (auto &Stmt : s)
    for (auto &Acc : Stmt)
      if (Acc->getType() == AccessTy) {
        isl::map rel = Acc->getLatestAccessRelation();
        rel = rel.intersect_domain(Stmt.getDomain());
        isl::space space = rel.get_space();
        space = space.range();
        space = space.from_range();
        space = space.set_tuple_id(isl::dim::in, Acc->getId());
        isl::map universe = isl::map::universe(space);
        rel = rel.domain_product(universe);
        if (rel.get_space().is_equal(targetSpace))
          return Acc;
      }

  LLVM_DEBUG(dbgs() << "schedule : " << schedule.to_str() << "\n");
  LLVM_DEBUG(dbgs() << "target Space :" << targetSpace.to_str() << "\n");
  assert(0 && "find not match in the memory accesses for the current space");
  return nullptr;
}

// remove unnecessary schedules.
// Why this function?
// Most of the time you have schedules as isl::union_map that look like:
// StmtY[i0, i1, i2] -> [(i0)]; StmtX[i0, i1] -> [(i0)]; StmtXlast[i0, i1] ->
// [(i0)] From this union_map you would like to extract only a single map. In
// the above case you would like to extract StmX[i0, i1] or StmtXlast[i0, i1]
// and check if the access pattern is an initialization stmt.
isl::union_map removeAdditionalSchedules(isl::union_map schedule,
                                         int expected_dim) {

  std::vector<isl::map> scheduleAsMaps;
  schedule.foreach_map([&](isl::map s) -> isl_stat {
    int dims = s.dim(isl::dim::in);
    if (dims != expected_dim)
      return isl_stat_ok;
    scheduleAsMaps.push_back(s);
    return isl_stat_ok;
  });

  if (scheduleAsMaps.size() > 1) {
    for (size_t i = 0; i < scheduleAsMaps.size(); i++) {
      for (size_t j = i + 1; j < scheduleAsMaps.size(); j++) {
        isl::id idOne =
            scheduleAsMaps[i].get_space().get_tuple_id(isl::dim::in);
        isl::id idTwo =
            scheduleAsMaps[j].get_space().get_tuple_id(isl::dim::in);
        std::string delimiter = "@";
        std::string sOne =
            idOne.to_str().substr(0, idOne.to_str().find(delimiter));
        std::string sTwo =
            idTwo.to_str().substr(0, idTwo.to_str().find(delimiter));
        if ((sOne.find(sTwo) != std::string::npos) ||
            (sTwo.find(sOne) != std::string::npos))
          scheduleAsMaps.erase(scheduleAsMaps.begin() + i);
      }
    }
  }

  // return the original schedule
  // if we were not able to remove
  // additional schedules.
  if (scheduleAsMaps.size() == 0)
    return schedule;

  isl::union_map res = isl::union_map(scheduleAsMaps[0]);
  for (size_t i = 1; i < scheduleAsMaps.size(); i++)
    res = res.unite(scheduleAsMaps[i]);

  return res;
}

/// Get a tagged access relation containing all accesses of type @p AccessTy.
/// Only array accesses are returned (see class MemoryAccess)
/// Instead of a normal access of the form:
///
///   Stmt[i,j,k] -> Array[f_0(i,j,k), f_1(i,j,k)]
///
/// a tagged access has the form
///
///   [Stmt[i,j,k] -> id[]] -> Array[f_0(i,j,k), f_1(i,j,k)]
///
/// where 'id' is an additional space that references the memory access that
/// triggered the access.
///
/// @param AccessTy The type of the memory accesses to collect.
///
/// @return The relation describing all tagged memory accesses.
isl::union_map getTaggedAccesses(const Scop &s,
                                 enum MemoryAccess::AccessType AccessTy) {

  isl::union_map accesses = isl::union_map::empty(s.getParamSpace());
  for (auto &stmt : s) {
    for (auto &acc : stmt) {
      if (acc->getLatestKind() != MemoryKind::Array)
        continue;
      if (acc->getType() == AccessTy) {
        isl::map relation = acc->getLatestAccessRelation();
        relation = relation.intersect_domain(stmt.getDomain());

        isl::space space = relation.get_space();
        space = space.range();
        space = space.from_range();
        space = space.set_tuple_id(isl::dim::in, acc->getId());
        isl::map universe = isl::map::universe(space);
        relation = relation.domain_product(universe);
        accesses = accesses.add_map(relation);
      }
    }
  }
  return accesses;
}

/// Get the set of all read accesses, tagged with access id.
isl::union_map getTaggedReads(const Scop &s) {
  return getTaggedAccesses(s, MemoryAccess::READ);
}

/// Get the set of all must write accesses, tagged with access id.
isl::union_map getTaggedMustWrites(const Scop &s) {
  return getTaggedAccesses(s, MemoryAccess::MUST_WRITE);
}

/// Given the schedule "schedule" returns
/// the tagged memory accesses that belong to schedule
/// and have the provided stmtId.
isl::union_map getTaggedReads(const Scop &s, isl::map schedule, isl::id stmtId) {

  isl::union_map res = isl::union_map::empty(schedule.get_space());
  isl::union_map taggedReads = getTaggedReads(s);
  // LLVM_DEBUG(dbgs() << "tagged reads : " << taggedReads.to_str() << "\n");
  std::vector<isl::map> vecTaggedReads;
  taggedReads.foreach_map([&](isl::map m) -> isl_stat {
    vecTaggedReads.push_back(m);
    return isl_stat_ok;
  });

  std::string idStmt = stmtId.to_str().substr(0, stmtId.to_str().find("@"));

  for (size_t i = 0; i < vecTaggedReads.size(); i++) {
    // FIXME do not use regex
    std::string mapAsString = vecTaggedReads[i].to_str();
    mapAsString.erase(
        std::remove_if(mapAsString.begin(), mapAsString.end(), isspace),
        mapAsString.end());
    std::smatch match;
    std::regex getStmtFromMap(R"(\{\[([a-z,A-Z,0-9,_]+))");
    std::regex_search(mapAsString, match, getStmtFromMap);
    if (match[1].compare(idStmt) == 0) {
      res = res.unite(isl::union_map(vecTaggedReads[i]));
    }
  }
  return res;
}

/// Given the schedule "schedule" returns
/// the tagged memory accesses that belong to schedule and have
/// the provided id.
isl::union_map getTaggedWrites(const Scop &s, isl::map schedule, isl::id stmtId) {

  isl::union_map res = isl::union_map::empty(schedule.get_space());
  isl::union_map taggedWrites = getTaggedMustWrites(s);
  std::vector<isl::map> vecTaggedWrites;
  taggedWrites.foreach_map([&](isl::map m) -> isl_stat {
    vecTaggedWrites.push_back(m);
    return isl_stat_ok;
  });

  std::string idStmt = stmtId.to_str().substr(0, stmtId.to_str().find("@"));

  for (size_t i = 0; i < vecTaggedWrites.size(); i++) {
    // FIXME do not use regex
    std::string mapAsString = vecTaggedWrites[i].to_str();
    mapAsString.erase(
        std::remove_if(mapAsString.begin(), mapAsString.end(), isspace),
        mapAsString.end());
    std::smatch match;
    std::regex getStmtFromMap(R"(\{\[([a-z,A-Z,0-9,_]+))");
    std::regex_search(mapAsString, match, getStmtFromMap);
    if (match[1].compare(idStmt) == 0) {
      res = res.unite(isl::union_map(vecTaggedWrites[i]));
    }
  }
  return res;
}

bool checkAccessGemmInitStmt(const Scop &s, isl::map schedule,
                             MatMulInfoTyExtended &MMI) {

  assert(MMI.A != nullptr && "Empty MMI.A");
  assert(MMI.B != nullptr && "Empty MMI.B");
  assert(MMI.ReadFromC != nullptr && "Empty MMI.ReadFromC");
  assert(MMI.WriteToC != nullptr && "Empty MMI.WriteToC");

  isl::union_map reads =
      getTaggedReads(s, schedule, schedule.get_tuple_id(isl::dim::in));
  isl::union_map writes =
      getTaggedWrites(s, schedule, schedule.get_tuple_id(isl::dim::in));

  isl::ctx ctx = s.getIslCtx();

  // expect one write.
  if (writes.n_map() != 1) {
    LLVM_DEBUG(dbgs() << "Expect at least one write access!\n");
    LLVM_DEBUG(dbgs() << "#writes: " << writes.n_map() << "\n");
    return false;
  }

  isl::map write = isl::map::from_union_map(writes);
  if (write.dim(isl::dim::in) != 2) {
    LLVM_DEBUG(dbgs() << "Schedule should have two dimensions!\n");
    LLVM_DEBUG(dbgs() << "#dims : " << write.dim(isl::dim::in) << "\n");
    return false;
  }

  using namespace matchers;
  using pMatches = Matches<SingleInputDim, FixedOutDimPattern<SimpleAff>>;
  using pGroup =
      PlaceholderGroupedSet<SingleInputDim, FixedOutDimPattern<SimpleAff>>;

  using namespace matchers;
  auto _i = placeholder(ctx);
  auto _ii = placeholder(ctx);
  auto _j = placeholder(ctx);
  auto _jj = placeholder(ctx);
  auto _A = arrayPlaceholder();

  pGroup psRead, psWrite;
  pMatches readMatches, writeMatches;
  std::vector<isl::space> iSpaceCandidates, jSpaceCandidates;
  std::vector<isl::space> iiSpaceCandidates, jjSpaceCandidates;

  bool hasRead = (reads.n_map() == 1);
  // we may not have the read access.
  // i.e., tmp[i][j] = 0;
  if (hasRead)
    psRead = allOf(access(_A, _i, _j));
  psWrite = allOf(access(_A, _ii, _jj));
  if (hasRead)
    readMatches = match(reads, psRead);
  writeMatches = match(writes, psWrite);

  if (hasRead && readMatches.size() != 1) {
    LLVM_DEBUG(
        dbgs() << "Read available but do not match the access pattern!\n");
    return false;
  }
  if (writeMatches.size() != 1) {
    LLVM_DEBUG(dbgs() << "Write do not match the access pattern!\n");
    return false;
  }

  if (hasRead) {
    if (writeMatches[0][_ii].payload().inputDimPos_ !=
        readMatches[0][_i].payload().inputDimPos_) {
      LLVM_DEBUG(dbgs() << "Index mismatch for i and ii\n");
      return false;
    }
    if (writeMatches[0][_jj].payload().inputDimPos_ !=
        readMatches[0][_j].payload().inputDimPos_) {
      LLVM_DEBUG(dbgs() << "Index mismatch for j and jj\n");
      return false;
    }
  }

  // if we have the read, collect the candidate spaces.
  if (hasRead) {
    iSpaceCandidates = readMatches[0][_i].candidateSpaces();
    jSpaceCandidates = readMatches[0][_j].candidateSpaces();
  }
  iiSpaceCandidates = writeMatches[0][_ii].candidateSpaces();
  jjSpaceCandidates = writeMatches[0][_jj].candidateSpaces();

  MMI.WriteToC_init = getMemoryAccessFromTagged(s, schedule, iiSpaceCandidates,
                                                jjSpaceCandidates, "w");
  if (hasRead)
    MMI.ReadFromC_init = getMemoryAccessFromTagged(
        s, schedule, iSpaceCandidates, jSpaceCandidates, "r");

  // make sure that MMI.WriteToC_init and MMI.WriteToC are pointing to the same
  // array.
  auto idAsStringWriteToC = MMI.WriteToC->getLatestArrayId().to_str();
  auto idAsStringWriteToC_init = MMI.WriteToC_init->getLatestArrayId().to_str();
  if (idAsStringWriteToC.compare(idAsStringWriteToC_init) != 0) {
    LLVM_DEBUG(dbgs() << "WriteToC != WriteToC_init\n");
    return false;
  }
  return true;
}

bool checkAccessGemmStmt(const Scop &s, isl::map schedule,
                         MatMulInfoTyExtended &MMI) {

  LLVM_DEBUG(
      dbgs() << "==========================================================\n");
  // s.dump();

  isl::ctx ctx = s.getIslCtx();
  isl::union_map reads =
      getTaggedReads(s, schedule, schedule.get_tuple_id(isl::dim::in));
  isl::union_map writes =
      getTaggedWrites(s, schedule, schedule.get_tuple_id(isl::dim::in));

  if (reads.n_map() != 3 || writes.n_map() != 1) {
    LLVM_DEBUG(dbgs() << "Expect 3 reads and 1 write\n");
    return false;
  }

  using namespace matchers;
  using pGroup =
      PlaceholderGroupedSet<SingleInputDim, FixedOutDimPattern<SimpleAff>>;
  auto _i = placeholder(ctx);
  auto _ii = placeholder(ctx);
  auto _j = placeholder(ctx);
  auto _jj = placeholder(ctx);
  auto _k = placeholder(ctx);
  auto _A = arrayPlaceholder();
  auto _B = arrayPlaceholder();
  auto _C = arrayPlaceholder();

  pGroup psReadNN;
  bool isNN = false;
  pGroup psReadNT;
  bool isNT = false;
  pGroup psReadTN;
  bool isTN = false;

  psReadNN = allOf(access(_A, _i, _j), access(_B, _i, _k), access(_C, _k, _j));
  psReadNT = allOf(access(_A, _i, _j), access(_B, _i, _k), access(_C, _j, _k));
  psReadTN = allOf(access(_A, _i, _j), access(_B, _k, _i), access(_C, _k, _j));
  // psRead = allOf(access(_A, _i, _j), access(_B, _k, _i), access(_C, _j, _k));

  auto psWrite = allOf(access(_A, _ii, _jj));

  auto readMatchesNN = match(reads, psReadNN);
  auto readMatchesNT = match(reads, psReadNT);
  auto readMatchesTN = match(reads, psReadTN);
  auto writeMatches = match(writes, psWrite);

  if (readMatchesNN.size() == 1) {
    if ((writeMatches[0][_ii].payload().inputDimPos_ ==
         readMatchesNN[0][_i].payload().inputDimPos_) &&
        (writeMatches[0][_jj].payload().inputDimPos_ ==
         readMatchesNN[0][_j].payload().inputDimPos_)) {
      isNN = true;
    } else {
      LLVM_DEBUG(
          dbgs() << "Index mismatch for i and ii or j and jj in gemmNN\n");
    }
  }

  if (readMatchesNT.size() == 1) {
    if ((writeMatches[0][_ii].payload().inputDimPos_ ==
         readMatchesNT[0][_i].payload().inputDimPos_) &&
        (writeMatches[0][_jj].payload().inputDimPos_ ==
         readMatchesNT[0][_j].payload().inputDimPos_)) {
      isNT = true;
    } else {
      LLVM_DEBUG(
          dbgs() << "Index mismatch for i and ii or j and jj in gemmNT\n");
    }
  }

  if (readMatchesTN.size() == 1) {
    if ((writeMatches[0][_ii].payload().inputDimPos_ ==
         readMatchesTN[0][_i].payload().inputDimPos_) &&
        (writeMatches[0][_jj].payload().inputDimPos_ ==
         readMatchesTN[0][_j].payload().inputDimPos_)) {
      isTN = true;
    } else {
      LLVM_DEBUG(
          dbgs() << "Index mismatch for i and ii or j and jj in gemmTN\n");
    }
  }

  if (isNN) {
    LLVM_DEBUG(dbgs() << "**** Access pattern GEMM-NN ****\n");
    LLVM_DEBUG(dbgs() << "_i  " << readMatchesNN[0][_i].payload().inputDimPos_
                      << "\n");
    LLVM_DEBUG(dbgs() << "_ii " << writeMatches[0][_ii].payload().inputDimPos_
                      << "\n");
    LLVM_DEBUG(dbgs() << "_j  " << readMatchesNN[0][_j].payload().inputDimPos_
                      << "\n");
    LLVM_DEBUG(dbgs() << "_jj " << writeMatches[0][_jj].payload().inputDimPos_
                      << "\n");
    LLVM_DEBUG(dbgs() << "_k  " << readMatchesNN[0][_k].payload().inputDimPos_
                      << "\n");
    LLVM_DEBUG(dbgs() << "******************************\n");
  }
  if (isNT) {
    LLVM_DEBUG(dbgs() << "**** Access pattern GEMM-NT ****\n");
    LLVM_DEBUG(dbgs() << "_i  " << readMatchesNT[0][_i].payload().inputDimPos_
                      << "\n");
    LLVM_DEBUG(dbgs() << "_ii " << writeMatches[0][_ii].payload().inputDimPos_
                      << "\n");
    LLVM_DEBUG(dbgs() << "_j  " << readMatchesNT[0][_j].payload().inputDimPos_
                      << "\n");
    LLVM_DEBUG(dbgs() << "_jj " << writeMatches[0][_jj].payload().inputDimPos_
                      << "\n");
    LLVM_DEBUG(dbgs() << "_k  " << readMatchesNT[0][_k].payload().inputDimPos_
                      << "\n");
    LLVM_DEBUG(dbgs() << "******************************\n");
  }
  if (isTN) {
    LLVM_DEBUG(dbgs() << "**** Access pattern GEMM-TN ****\n");
    LLVM_DEBUG(dbgs() << "_i  " << readMatchesTN[0][_i].payload().inputDimPos_
                      << "\n");
    LLVM_DEBUG(dbgs() << "_ii " << writeMatches[0][_ii].payload().inputDimPos_
                      << "\n");
    LLVM_DEBUG(dbgs() << "_j  " << readMatchesTN[0][_j].payload().inputDimPos_
                      << "\n");
    LLVM_DEBUG(dbgs() << "_jj " << writeMatches[0][_jj].payload().inputDimPos_
                      << "\n");
    LLVM_DEBUG(dbgs() << "_k  " << readMatchesTN[0][_k].payload().inputDimPos_
                      << "\n");
    LLVM_DEBUG(dbgs() << "******************************\n");
  }

  LLVM_DEBUG(dbgs() << "isNN :" << isNN << "\n");
  LLVM_DEBUG(dbgs() << "isNT :" << isNT << "\n");
  LLVM_DEBUG(dbgs() << "isTN :" << isTN << "\n");

  if (isNN == true && (isNT == true || isTN == true))
    assert(0 && "isNN true and also isNT or is TN");
  if (isNT == true && (isNN == true || isTN == true))
    assert(0 && "isNT true and also isNN or isTN");
  if (isTN == true && (isNN == true || isNT == true))
    assert(0 && "isTN true and also isNN or isNT");

  if (isNN == false && isNT == false && isTN == false)
    return false;

  if (writeMatches.size() != 1)
    return false;

  if (isNN) {
    LLVM_DEBUG(dbgs() << "**** Access pattern GEMM-NN ****\n");
    LLVM_DEBUG(dbgs() << "_i  " << readMatchesNN[0][_i].payload().inputDimPos_
                      << "\n");
    LLVM_DEBUG(dbgs() << "_ii " << writeMatches[0][_ii].payload().inputDimPos_
                      << "\n");
    LLVM_DEBUG(dbgs() << "_j  " << readMatchesNN[0][_j].payload().inputDimPos_
                      << "\n");
    LLVM_DEBUG(dbgs() << "_jj " << writeMatches[0][_jj].payload().inputDimPos_
                      << "\n");
    LLVM_DEBUG(dbgs() << "_k  " << readMatchesNN[0][_k].payload().inputDimPos_
                      << "\n");
    LLVM_DEBUG(dbgs() << "******************************\n");
  }
  if (isNT) {
    LLVM_DEBUG(dbgs() << "**** Access pattern GEMM-NT ****\n");
    LLVM_DEBUG(dbgs() << "_i  " << readMatchesNT[0][_i].payload().inputDimPos_
                      << "\n");
    LLVM_DEBUG(dbgs() << "_ii " << writeMatches[0][_ii].payload().inputDimPos_
                      << "\n");
    LLVM_DEBUG(dbgs() << "_j  " << readMatchesNT[0][_j].payload().inputDimPos_
                      << "\n");
    LLVM_DEBUG(dbgs() << "_jj " << writeMatches[0][_jj].payload().inputDimPos_
                      << "\n");
    LLVM_DEBUG(dbgs() << "_k  " << readMatchesNT[0][_k].payload().inputDimPos_
                      << "\n");
    LLVM_DEBUG(dbgs() << "******************************\n");
  }
  if (isTN) {
    LLVM_DEBUG(dbgs() << "**** Access pattern GEMM-TN ****\n");
    LLVM_DEBUG(dbgs() << "_i  " << readMatchesTN[0][_i].payload().inputDimPos_
                      << "\n");
    LLVM_DEBUG(dbgs() << "_ii " << writeMatches[0][_ii].payload().inputDimPos_
                      << "\n");
    LLVM_DEBUG(dbgs() << "_j  " << readMatchesTN[0][_j].payload().inputDimPos_
                      << "\n");
    LLVM_DEBUG(dbgs() << "_jj " << writeMatches[0][_jj].payload().inputDimPos_
                      << "\n");
    LLVM_DEBUG(dbgs() << "_k  " << readMatchesTN[0][_k].payload().inputDimPos_
                      << "\n");
    LLVM_DEBUG(dbgs() << "******************************\n");
  }

  std::vector<isl::space> iSpaceCandidates;
  std::vector<isl::space> iiSpaceCandidates;
  std::vector<isl::space> jSpaceCandidates;
  std::vector<isl::space> jjSpaceCandidates;
  std::vector<isl::space> kSpaceCandidates;

  if (isNN) {
    iSpaceCandidates = readMatchesNN[0][_i].candidateSpaces();
    iiSpaceCandidates = writeMatches[0][_ii].candidateSpaces();
    jSpaceCandidates = readMatchesNN[0][_j].candidateSpaces();
    jjSpaceCandidates = writeMatches[0][_jj].candidateSpaces();
    kSpaceCandidates = readMatchesNN[0][_k].candidateSpaces();
  }
  if (isNT) {
    iSpaceCandidates = readMatchesNT[0][_i].candidateSpaces();
    iiSpaceCandidates = writeMatches[0][_ii].candidateSpaces();
    jSpaceCandidates = readMatchesNT[0][_j].candidateSpaces();
    jjSpaceCandidates = writeMatches[0][_jj].candidateSpaces();
    kSpaceCandidates = readMatchesNT[0][_k].candidateSpaces();
  }
  if (isTN) {
    iSpaceCandidates = readMatchesTN[0][_i].candidateSpaces();
    iiSpaceCandidates = writeMatches[0][_ii].candidateSpaces();
    jSpaceCandidates = readMatchesTN[0][_j].candidateSpaces();
    jjSpaceCandidates = writeMatches[0][_jj].candidateSpaces();
    kSpaceCandidates = readMatchesTN[0][_k].candidateSpaces();
  }

  MMI.WriteToC = getMemoryAccessFromTagged(s, schedule, iiSpaceCandidates,
                                           jjSpaceCandidates, "w");
  MMI.ReadFromC = getMemoryAccessFromTagged(s, schedule, iSpaceCandidates,
                                            jSpaceCandidates, "r");
  MMI.A = getMemoryAccessFromTagged(s, schedule, iSpaceCandidates,
                                    kSpaceCandidates, "r");
  MMI.B = getMemoryAccessFromTagged(s, schedule, kSpaceCandidates,
                                    jSpaceCandidates, "r");

  if (isNN) {
    MMI.isTransposeA = false;
    MMI.isTransposeB = false;
  }
  if (isNT) {
    MMI.isTransposeA = false;
    MMI.isTransposeB = true;
  }
  if (isTN) {
    MMI.isTransposeA = true;
    MMI.isTransposeB = false;
  }

  LLVM_DEBUG(
      dbgs() << "==========================================================\n");
  return true;
}

/// Apply the tiling transformation.
std::pair<isl::multi_union_pw_aff, isl::multi_union_pw_aff>
tile_node(isl::schedule_node node, int tileSize) {

  auto space = isl::manage(isl_schedule_node_band_get_space(node.get()));
  auto dims = space.dim(isl::dim::set);
  auto sizes = isl::multi_val::zero(space);

  for (unsigned i = 0; i < dims; i++) {
    sizes = sizes.set_val(i, isl::val(node.get_ctx(), tileSize));
  }

  node =
      isl::manage(isl_schedule_node_band_tile(node.release(), sizes.release()));
  auto res = std::make_pair(node.band_get_partial_schedule(),
                            node.child(0).band_get_partial_schedule());
  return res;
}

// fw. decl.
isl::schedule fuseTwoConsecutiveGemmIfNotTiled(isl::schedule schedule, const Scop &s);

static isl::schedule_node addCimStartUp(isl::schedule_node node) {

  isl::space space;
  isl::union_set domain;
  isl::schedule_node graft;

  space = isl::space(node.get_ctx(), 0 ,0);
  space = space.set_tuple_name(isl::dim::set, "cim_init");  
  domain = isl::union_set(isl::set::universe(space));
  graft = isl::schedule_node::from_domain(domain);
  node = node.graft_before(graft);
  return node;
}

static isl::schedule_node addCimAllocateSharedMemory(
  isl::schedule_node node, int bytes) {

  isl::space space;
  isl::union_set domain;
  isl::schedule_node graft;

  space = isl::space(node.get_ctx(), 0 ,0);
  space = space.set_tuple_name(isl::dim::set, "cim_allocate_shared_memory");  
  domain = isl::union_set(isl::set::universe(space));
  graft = isl::schedule_node::from_domain(domain);
  int *p_bytes = new int;
  *p_bytes = bytes;
  graft = graft.child(0).
    insert_mark(isl::id::alloc(node.get_ctx(), "__cim_allocate_", p_bytes));
  graft = graft.parent();
  node = node.graft_before(graft);
  return node; 

}

static isl::schedule_node addCimTearDown(isl::schedule_node node) {

  isl::space space;
  isl::union_set domain;
  isl::schedule_node graft;

  space = isl::space(node.get_ctx(), 0 ,0);
  space = space.set_tuple_name(isl::dim::set, "cim_tear_down");
  domain = isl::union_set(isl::set::universe(space));
  graft = isl::schedule_node::from_domain(domain);
  node = node.graft_after(graft);
  return node;
}

// is the pattern gemm-like?
isl::schedule isGemmLikeLate(isl::schedule schedule, const Scop &s, const Tactic tac) {

  isl::schedule_node root = schedule.get_root();

  // see ScheduleOptimizer.h
  pGemm.flush();

  // check gemm conditions:
  // 1. We mast have a single-map schedule
  // 2. The input dimension for the schedule must be >= 3
  auto hasGemmConditions = [&](isl::schedule_node band) {
    isl::union_map sched =
        isl::union_map::from(band.band_get_partial_schedule());
    if (sched.n_map() != 1) {
      LLVM_DEBUG(dbgs() << "hasGemmConditions: false due to n_map != 1\n");
      LLVM_DEBUG(dbgs() << "number of map is: " << sched.n_map() << "\n");
      LLVM_DEBUG(dbgs() << band.to_str() << "\n");
      return false;
    }
    isl::map schedAsMap = isl::map::from_union_map(sched);
    if (schedAsMap.dim(isl::dim::out) < 3) {
      LLVM_DEBUG(dbgs() << "hasGemmConditions: false due to in dim < 3\n");
      LLVM_DEBUG(dbgs() << "output dimension is :"
                        << schedAsMap.dim(isl::dim::in) << "\n");
      LLVM_DEBUG(dbgs() << band.to_str() << "\n");
      return false;
    }
    LLVM_DEBUG(dbgs() << "hasGemmConditions: true\n");
    return true;
  };

  // check gemm access pattern.
  auto containsMatrMul = [&](isl::schedule_node band) {
    MatMulInfoTyExtended MMI;

    isl::map scheduleAsMap = isl::map::from_union_map(
        isl::union_map::from(band.band_get_partial_schedule()));

    if (!checkAccessGemmStmt(s, scheduleAsMap, MMI)) {
      LLVM_DEBUG(
          dbgs() << "containsMatrMul: false due to checkAccessGemmStmt\n");
      return false;
    }
    pGemm.detected++;
    pGemm.patternTys.push_back(MMI);
    LLVM_DEBUG(dbgs() << "containsMatrMul: true\n");
    return true;
  };

  // This callback avoids entering an infinite loop
  // during recursion (wrapPatternDFSPreorder).
  // Specifically, the callback checks if the matcher
  // already fired.
  auto hasNotFired = [&](isl::schedule_node band) {
    if (!band.has_parent())
      return true;
    auto maybeMark = band.parent();
    if (isl_schedule_node_get_type(maybeMark.get()) != isl_schedule_node_mark)
      return true;
    auto markId = maybeMark.mark_get_id().to_str();
    markId = markId.substr(0, markId.find("@"));
    if ((markId.compare("gemm") == 0) || (markId.compare("gemm_init") == 0))
      return false;
    else
      return true;
  };
/*
  // This callback always returns true. The only purpose is to
  // check if tiling is profitable for the CIM device.
  auto needToTile = [&](isl::schedule_node band) {
    auto schedule = band.get_prefix_schedule_union_map();
    schedule = schedule.intersect_domain(s.getDomains());
    schedule.foreach_map([&](isl::map m) {
      auto max_i = m.domain().dim_max(0);
      auto max_j = m.domain().dim_max(1);
      auto max_k = m.domain().dim_max(2);

      isl::val val_max_i;
      isl::val val_max_j;
      isl::val val_max_k;
      max_i.foreach_piece([&](isl::set s, isl::aff a) -> isl_stat {
        val_max_i = a.get_constant_val();
        return isl_stat_ok;
      });
      max_j.foreach_piece([&](isl::set s, isl::aff a) -> isl_stat {
        val_max_j = a.get_constant_val();
        return isl_stat_ok;
      });
      max_k.foreach_piece([&](isl::set s, isl::aff a) -> isl_stat {
        val_max_k = a.get_constant_val();
        return isl_stat_ok;
      });
      LLVM_DEBUG(dbgs() << val_max_i.to_str() << "\n");
      LLVM_DEBUG(dbgs() << val_max_j.to_str() << "\n");
      LLVM_DEBUG(dbgs() << val_max_k.to_str() << "\n");
      if (std::stoi(val_max_i.to_str()) < TILE_FACTOR_CIM_DEVICE ||
          std::stoi(val_max_j.to_str()) < TILE_FACTOR_CIM_DEVICE ||
          std::stoi(val_max_k.to_str()) < TILE_FACTOR_CIM_DEVICE) {
        LLVM_DEBUG(dbgs() << "set to false\n");
      }
      return isl_stat_ok;
    });
    return true;
  }; 
*/
  // look for the gemm pattern
  isl::schedule_node gemm_body;
  auto matcherGemm = [&]() {
    using namespace matchers;
    // clang-format off
    return
      band(_and(
          hasNotFired, hasGemmConditions, containsMatrMul/*, needToTile*/), gemm_body,
        leaf());
    // clang-format on
  }();

  // rebuild gemm pattern with a fixed tile size,
  // needed by the cim device. 
  auto builderGemm = builders::ScheduleNodeBuilder();
  {
    using namespace builders;

    auto computeScheduleTile = [&]() {
      auto descr = BandDescriptor(gemm_body);
      auto tiled_schedule = tile_node(gemm_body, TILE_FACTOR_CIM_DEVICE);
      descr.partialSchedule = tiled_schedule.first;
      return descr;
    };
    auto computeSchedulePoint = [&]() {
      auto descr = BandDescriptor(gemm_body);
      auto tiled_schedule = tile_node(gemm_body, TILE_FACTOR_CIM_DEVICE);
      descr.partialSchedule = tiled_schedule.second;
      return descr;
    };
    auto marker = [&]() {
      return isl::id::alloc(s.getIslCtx(), "gemm", &pGemm);
    };
    auto originalSchedule = [&]() {
      auto descr = BandDescriptor(gemm_body);
      return descr; 
    };
    builderGemm = (tac == Tactic::TILING) ?
      band(computeScheduleTile, mark(marker, band(computeSchedulePoint))) :
      mark(marker, band(originalSchedule));
  }

  root = replaceDFSPreorderOnce(root.child(0), matcherGemm, builderGemm);

  if (tac == Tactic::FUSION) {
    root = 
      fuseTwoConsecutiveGemmIfNotTiled(root.root().get_schedule(), s).get_root();
  }

  // early exit if we did not detect any core gemm stmt.
  // if we did detect a gemm pattern we also look for
  // a possible initialization stmt.
  // FIXME: Kanishkan are we also interested in the detecting
  // init stmt for gemm? If not we early exit here and drop
  // matcherInit.
  if (!lookUpScheduleTree(root.root().get_schedule(), "gemm")) {
    return root.root().get_schedule();
  }

  // check conditions for init stmt.
  // 1. single-map schedule after removed redundant schedules.
  auto hasGemmInitConditions = [&](isl::schedule_node band) {
    isl::union_map sched =
        isl::union_map::from(band.band_get_partial_schedule());
    sched = removeAdditionalSchedules(sched, 2);
    if (sched.n_map() != 1) {
      LLVM_DEBUG(dbgs() << "hasGemmInitConditions: false due to n_map != 1\n");
      return false;
    }
    LLVM_DEBUG(dbgs() << "hasGemmInitConditions: true\n");
    return true;
  };

  // check the access pattern for the init stmt that must
  // be compliant with the gemm one.
  auto containsMatrMulInit = [&](isl::schedule_node band) {
    isl::union_map sched =
        isl::union_map::from(band.band_get_partial_schedule());
    sched = removeAdditionalSchedules(sched, 2);
    isl::map scheduleAsMap = isl::map::from_union_map(sched);
    if (!checkAccessGemmInitStmt(s, scheduleAsMap,
                                 pGemm.patternTys[pGemm.current])) {
      LLVM_DEBUG(
          dbgs()
          << "containsMatrMulInit: false due to checkAccessGemmInitStmt\n");
      return false;
    }
    LLVM_DEBUG(dbgs() << "containsMatrMulInit: true\n");
    pGemm.current++;
    return true;
  };

  // check if the init stmt dominates and is in the same
  // subtree of the gemm pattern.
  // NOTE: we may lose some potential optimization. We may
  // not find the init stmt if this rule does not apply.
  auto hasAnnotation = [&](isl::schedule_node mark) {
    isl::id markId = mark.mark_get_id();
    std::string idAsString = markId.to_str();
    idAsString = idAsString.substr(0, idAsString.find("@"));
    if (idAsString.compare("gemm") == 0) {
      LLVM_DEBUG(dbgs() << "hasAnnotation -> true\n");
      return true;
    }
    LLVM_DEBUG(dbgs() << "hasAnnotation -> false\n");
    return false;
  };

  isl::schedule_node subtree;
  isl::schedule_node dummySubtree;
  auto matcherInit = [&]() {
    using namespace matchers;
    // clang-format off
    return
      band(_and(hasChild(mark(hasAnnotation, band(anyTree(dummySubtree)))), 
                hasNotFired,
                hasGemmInitConditions, 
                containsMatrMulInit),
        anyTree(subtree));
    // clang-format on
  }();

  root = wrapPatternDFSPreorder(s.getIslCtx(), &pGemm, "gemm_init", root,
                                matcherInit);
  pGemm.current = 0;
  return root.root().get_schedule();
}

static bool checkFusionHelper(std::string a_x, std::string b_x,
  std::string c_x, std::string a_y, std::string b_y, std::string c_y) {

  if (c_y.compare(c_x) == 0)
    return false;
  if ((a_y.compare(c_x) == 0) || (b_y.compare(c_x) == 0))
    return false;
  return true;
}

static bool checkFusion() {

  assert(pGemm.patternTys.size() == 2);
  
  ScopArrayInfo *sai_a_x =
    const_cast<ScopArrayInfo *>(
      pGemm.patternTys[0].A->getLatestScopArrayInfo());
  ScopArrayInfo *sai_b_x =
    const_cast<ScopArrayInfo *>(
      pGemm.patternTys[0].B->getLatestScopArrayInfo());
  ScopArrayInfo *sai_c_x =
    const_cast<ScopArrayInfo *>(
      pGemm.patternTys[0].WriteToC->getLatestScopArrayInfo());

  ScopArrayInfo *sai_a_y =
    const_cast<ScopArrayInfo *>(
      pGemm.patternTys[1].A->getLatestScopArrayInfo());
  ScopArrayInfo *sai_b_y =
    const_cast<ScopArrayInfo *>(
      pGemm.patternTys[1].B->getLatestScopArrayInfo());
  ScopArrayInfo *sai_c_y =
    const_cast<ScopArrayInfo *>(
      pGemm.patternTys[1].WriteToC->getLatestScopArrayInfo());

  std::string a_x = sai_a_x->getName();
  std::string b_x = sai_b_x->getName(); 
  std::string c_x = sai_c_x->getName();
  std::string a_y = sai_a_y->getName();
  std::string b_y = sai_b_y->getName();
  std::string c_y = sai_c_y->getName();

  return checkFusionHelper(a_x, b_x, c_x, a_y, b_y, c_y);
}


// First attemp to kernel fusion.
// 
// We fuse two *consecutive* gemm pattern
// if *not tiled*.
// In addition, we fuse only if the two kernels are
// idependent, meaning that the following condition 
// should hold:
// 1. Given to fusion candidates X and Y with Y following *directly* X
// Y must not read from or write to any output of X and does not
// write to any input of Y.
// This function **must** be called after isGemmLikeLate
// TODO: Kanishkan: We need to have a cost function for fusion for the CIM
// device. Any idea? 
isl::schedule fuseTwoConsecutiveGemmIfNotTiled
(isl::schedule schedule, const Scop &s) {

  // early exit if the number of gemm pattern detected 
  // is less than two.
  if (pGemm.patternTys.size() < 2) {
    return schedule;
  }

  bool is_fusion_valid = checkFusion();
  if (!is_fusion_valid)
    return schedule;

  isl::schedule_node root = schedule.get_root();

  auto hasGemmId = [&](isl::schedule_node mark) {
    
    auto mark_id = mark.mark_get_id().to_str();
    mark_id = mark_id.substr(0, mark_id.find("@"));
    if (!mark_id.compare("gemm") == 0) {
      return false;
    }
    return true;
  };

  isl::schedule_node domain_node;
  isl::schedule_node filter_node_upper, filter_node_lower;
  isl::schedule_node schedule_node_upper, schedule_node_lower;
  auto matcher = [&]() {
    // clang-forma off
    using namespace matchers;
    return domain(
      domain_node,
        sequence(filter(filter_node_upper, 
                   mark(hasGemmId, band(schedule_node_upper, leaf()))),
                 filter(filter_node_lower, 
                   mark(hasGemmId, band(schedule_node_lower, leaf())))));
  }();
  // clang-format on

  if (!matchers::ScheduleNodeMatcher::isMatching(matcher, root))
    return schedule;

  auto fused_schedule = schedule_node_upper.band_get_partial_schedule();
  fused_schedule =
      fused_schedule.union_add(schedule_node_lower.band_get_partial_schedule());

  // XXX: we cannot cut in between sequence, so we need
  // to recompute the *entire* tree. This limits the applicability
  // of this transformation.
  auto new_root = [&]() {
    using namespace builders;
    // clang-format off
    auto builder =
        domain(domain_node.domain_get_domain(),
               band(fused_schedule,
                    sequence(filter(filter_node_upper.filter_get_filter()),
                             filter(filter_node_lower.filter_get_filter()))));  
    // clang-format on
    return builder.build();
  }();

  return new_root.get_schedule();
}

bool checkAccessGemvStmt(const Scop &s, isl::map schedule, MatVecInfoTy &MVI) {

  isl::ctx ctx = s.getIslCtx();

  isl::union_map reads =
      getTaggedReads(s, schedule, schedule.get_tuple_id(isl::dim::in));
  isl::union_map writes =
      getTaggedWrites(s, schedule, schedule.get_tuple_id(isl::dim::in));

  if (reads.n_map() != 3 || writes.n_map() != 1) {
    LLVM_DEBUG(dbgs() << "Expect 3 reads and 1 write for gemv!\n");
    LLVM_DEBUG(dbgs() << "reads : " << reads.to_str() << "\n");
    LLVM_DEBUG(dbgs() << "Writes : " << writes.to_str() << "\n");
    LLVM_DEBUG(dbgs() << "#reads : " << reads.n_map() << "\n");
    LLVM_DEBUG(dbgs() << "#writes : " << writes.n_map() << "\n");
    return false;
  }

  // check additional condition on write. Expect access to
  // 1D vector in a 2d nested loop.
  isl::map writeAsMap = isl::map::from_union_map(writes);
  if (writeAsMap.dim(isl::dim::in) != 2 || writeAsMap.dim(isl::dim::out) != 1) {
    LLVM_DEBUG(dbgs() << "Expect a write access to 1D array in a 2d nest!\n");
    return false;
  }

  using namespace matchers;
  auto _i = placeholder(ctx);
  auto _ii = placeholder(ctx);
  auto _j = placeholder(ctx);
  auto _A = arrayPlaceholder();
  auto _B = arrayPlaceholder();
  auto _C = arrayPlaceholder();
  auto psRead = allOf(access(_A, _i, _j), access(_B, _j), access(_C, _i));
  auto psWrite = allOf(access(_C, _ii));
  auto readMatches = match(reads, psRead);
  auto writeMatches = match(writes, psWrite);

  if (readMatches.size() != 1) {
    LLVM_DEBUG(dbgs() << "readMatches.size() != 1\n");
    LLVM_DEBUG(dbgs() << readMatches.size() << "\n");
    LLVM_DEBUG(dbgs() << "reads : " << reads.to_str() << "\n");
    LLVM_DEBUG(dbgs() << "Writes : " << writes.to_str() << "\n");
    return false;
  }
  if (writeMatches.size() != 1) {
    LLVM_DEBUG(dbgs() << "writeMatches.size() != 1\n");
    LLVM_DEBUG(dbgs() << "reads : " << reads.to_str() << "\n");
    LLVM_DEBUG(dbgs() << "Writes : " << writes.to_str() << "\n");
    return false;
  }

  auto iSpaceCandidates = readMatches[0][_i].candidateSpaces();
  auto jSpaceCandidates = readMatches[0][_j].candidateSpaces();
  auto iiSpaceCandidates = writeMatches[0][_ii].candidateSpaces();

  if (writeMatches[0][_ii].payload().inputDimPos_ ==
      readMatches[0][_i].payload().inputDimPos_) {
    LLVM_DEBUG(dbgs() << "Access to matrix is non transposed\n");
    LLVM_DEBUG(dbgs() << "for (i\n");
    LLVM_DEBUG(dbgs() << "  for(j\n");
    LLVM_DEBUG(dbgs() << "    x[i] = ... A[i][otherDim]\n");
    MVI.isTranspose = false;
  }

  if (writeMatches[0][_ii].payload().inputDimPos_ ==
      readMatches[0][_j].payload().inputDimPos_) {
    LLVM_DEBUG(dbgs() << "Access to matrix is transposed\n");
    LLVM_DEBUG(dbgs() << "for (i\n");
    LLVM_DEBUG(dbgs() << "  for(j\n");
    LLVM_DEBUG(dbgs() << "    x[i] = ... A[otherDim][i]\n");
    MVI.isTranspose = true;
  }

  if ((writeMatches[0][_ii].payload().inputDimPos_ !=
       readMatches[0][_i].payload().inputDimPos_) &&
      (writeMatches[0][_ii].payload().inputDimPos_ !=
       readMatches[0][_j].payload().inputDimPos_)) {
    return false;
  }

  MVI.A = getMemoryAccessFromTagged(s, schedule, iSpaceCandidates,
                                    jSpaceCandidates, "r");
  MVI.WriteToY = getMemoryAccessFromTagged(s, schedule, iiSpaceCandidates,
                                           iiSpaceCandidates, "w");

  MVI.ReadFromY = (MVI.isTranspose)
                      ? getMemoryAccessFromTagged(s, schedule, jSpaceCandidates,
                                                  jSpaceCandidates, "r")
                      : getMemoryAccessFromTagged(s, schedule, iSpaceCandidates,
                                                  iSpaceCandidates, "r");

  MVI.X = (MVI.isTranspose)
              ? getMemoryAccessFromTagged(s, schedule, iSpaceCandidates,
                                          iSpaceCandidates, "r")
              : getMemoryAccessFromTagged(s, schedule, jSpaceCandidates,
                                          jSpaceCandidates, "r");
  return true;
}

// is the patter matrix-vector like?
static isl::schedule isGemvLikeLate(isl::schedule schedule, const Scop &s) {

  isl::schedule_node root = schedule.get_root();
  
   pGemv.flush();

  auto hasGemvConditions = [&](isl::schedule_node band) {
    isl::union_map schedule =
        isl::union_map::from(band.band_get_partial_schedule());
    if (schedule.n_map() != 1)
      return false;
    isl::map scheduleAsMap = isl::map::from_union_map(schedule);
    if (scheduleAsMap.dim(isl::dim::out) != 2)
      return false;
    return true;
  };

  // check if the access patterns are gemv like.
  auto containsGemv = [&](isl::schedule_node band) {
    MatVecInfoTy MVI;
    isl::map scheduleAsMap = isl::map::from_union_map(
        isl::union_map::from(band.band_get_partial_schedule()));
    if (!checkAccessGemvStmt(s, scheduleAsMap, MVI))
      return false;
    pGemv.detected++;
    pGemv.patternTys.push_back(MVI);
    return true;
  };

  auto hasNotFired = [&](isl::schedule_node band) {
    if (!band.has_parent())
      return true;
    auto maybeMark = band.parent();
    if (isl_schedule_node_get_type(maybeMark.get()) != isl_schedule_node_mark)
      return true;
    auto markId = maybeMark.mark_get_id().to_str();
    markId = markId.substr(0, markId.find("@"));
    if (markId.compare("gemv") == 0)
      return false;
    else
      return true;
  };

  isl::schedule_node gemvBody, subtree;
  auto matcherGemv = [&]() {
    using namespace matchers;
    // clang-format off
    return band(_and(hasNotFired,
                     hasGemvConditions, 
                     containsGemv), gemvBody, anyTree(subtree));
    // clang-format on
  }();

  // rebuild gemm pattern with a fixed tile size,
  // needed by the cim device.
  auto builderGemv = builders::ScheduleNodeBuilder();
  {
    using namespace builders;

    auto computeScheduleTile = [&]() {
      auto descr = BandDescriptor(gemvBody);
      auto tiledSchedule = tile_node(gemvBody, TILE_FACTOR_CIM_DEVICE);
      descr.partialSchedule = tiledSchedule.first;
      return descr;
    };
    auto computeSchedulePoint = [&]() {
      auto descr = BandDescriptor(gemvBody);
      auto tiledSchedule = tile_node(gemvBody, TILE_FACTOR_CIM_DEVICE);
      descr.partialSchedule = tiledSchedule.second;
      return descr;
    };
    auto marker = [&]() {
      return isl::id::alloc(s.getIslCtx(), "gemv", &pGemv);
    };
    builderGemv = 
      band(computeScheduleTile, mark(marker, band(computeSchedulePoint)));
  }

  root = replaceDFSPreorderOnce(root.child(0), matcherGemv, builderGemv);
  return root.root().get_schedule();
}

template<class T, class...>
struct are_same : std::true_type
{};

template<class T, class U, class... TT>
struct are_same<T, U, TT...>
    : std::integral_constant<bool, std::is_same<T,U>{} && are_same<T, TT...>{}>
{};

template <typename T, typename arg, typename... args>
void do_for(T f, arg first, args... rest) {

  f(first);
  do_for(f, rest...);
}

template <typename T>
void do_for(T f) {}

/// Return the array extent.
static isl::set getArrayExtent(const ScopArrayInfo *Array, const Scop &S) {
  unsigned NumDims = Array->getNumberOfDimensions();

  if (Array->getNumberOfDimensions() == 0)
    return isl::set::universe(Array->getSpace());

  //XXX
  isl::union_map Accesses = 
    const_cast<Scop&>(S).getAccesses(const_cast<ScopArrayInfo*>(Array));
  isl::union_set AccessUSet = Accesses.range();
  AccessUSet = AccessUSet.coalesce();
  AccessUSet = AccessUSet.detect_equalities();
  AccessUSet = AccessUSet.coalesce();

  if (AccessUSet.is_empty())
    return isl::set::empty(Array->getSpace());

  isl::set AccessSet = AccessUSet.extract_set(Array->getSpace());

  isl::local_space LS = isl::local_space(Array->getSpace());

  isl::pw_aff Val = isl::aff::var_on_domain(LS, isl::dim::set, 0);
  isl::pw_aff OuterMin = AccessSet.dim_min(0);
  isl::pw_aff OuterMax = AccessSet.dim_max(0);
  OuterMin = OuterMin.add_dims(isl::dim::in, Val.dim(isl::dim::in));
  OuterMax = OuterMax.add_dims(isl::dim::in, Val.dim(isl::dim::in));
  OuterMin = OuterMin.set_tuple_id(isl::dim::in, Array->getBasePtrId());
  OuterMax = OuterMax.set_tuple_id(isl::dim::in, Array->getBasePtrId());

  isl::set Extent = isl::set::universe(Array->getSpace());

  Extent = Extent.intersect(OuterMin.le_set(Val));
  Extent = Extent.intersect(OuterMax.ge_set(Val));

  for (unsigned i = 1; i < NumDims; ++i)
    Extent = Extent.lower_bound_si(isl::dim::set, i, 0);

  for (unsigned i = 0; i < NumDims; ++i) {
    isl::pw_aff PwAff = Array->getDimensionSizePw(i);

    // isl_pw_aff can be NULL for zero dimension. Only in the case of a
    // Fortran array will we have a legitimate dimension.
    if (PwAff.is_null()) {
      assert(i == 0 && "invalid dimension isl_pw_aff for nonzero dimension");
      continue;
    }
    isl::pw_aff Val = isl::aff::var_on_domain(
        isl::local_space(Array->getSpace()), isl::dim::set, i);
    PwAff = PwAff.add_dims(isl::dim::in, Val.dim(isl::dim::in));
    PwAff = PwAff.set_tuple_id(isl::dim::in, Val.get_tuple_id(isl::dim::in));
    isl::set Set = PwAff.gt_set(Val);
    Extent = Set.intersect(Extent);
  }

  return Extent;
}

/// Return lower and upper bound for a given dimension
/// of a given array.
/// i.e., A[i][j] : 0 <= i <= 1024, 0 <= j <= 2048
/// getDimensionBounds(A, 0) returns:
/// -> 0 as lower bound
/// -> 1024 as upper bound
///
/// If we cannot statically compute the array bounds the
/// function returns (-1, -1)
static std::pair<isl::val, isl::val>
getDimensionBounds(isl::ctx ctx, isl::set extent, int dim) {

  assert(static_cast<size_t>(dim) < extent.dim(isl::dim::set) &&
         "must be less!\n");

  isl::pw_aff lower_bound = extent.dim_min(dim);
  isl::pw_aff upper_bound = extent.dim_max(dim);

  assert(lower_bound.n_piece() == 1 && "expect single piece");
  assert(upper_bound.n_piece() == 1 && "expect single piece");

  isl::val lower_bound_val;
  isl::val upper_bound_val;
  lower_bound.foreach_piece([&](isl::set s, isl::aff a) -> isl_stat {
    lower_bound_val = a.get_constant_val();
    return isl_stat_ok;
  });
  upper_bound.foreach_piece([&](isl::set s, isl::aff a) -> isl_stat {
    upper_bound_val = a.get_constant_val();
    return isl_stat_ok;
  });

  upper_bound_val = upper_bound_val.add(isl::val::one(ctx));
  return std::make_pair(lower_bound_val, upper_bound_val);
}



/// Get the dimensions bounds for a 2d array.
/// i.e., A[i][j] : 0 <= i <= 1024, 0 <= j <= 2048
/// getArrayBounds(A) returns:
/// -> 0 as lower bound for dimension i
/// -> 1024 as upper bound for dimension i
/// -> 0 as lower bound for dimension j
/// -> 2048 as upper bound for dimension j
static std::tuple<isl::val, isl::val, isl::val, isl::val> 
getArrayBounds(const ScopArrayInfo *Array, const Scop &S) {

  isl::ctx ctx = Array->getScop().getIslCtx();
  isl::val negone = isl::val::negone(ctx);

  if (!Array->isArrayKind()) {
    return std::make_tuple(negone, negone, negone, negone);
  }

  size_t dims = Array->getNumberOfDimensions();
  if (dims != 2) {
    LLVM_DEBUG(dbgs() << "Expect 2d arrays only!\n");
    return std::make_tuple(negone, negone, negone, negone);
  }

  isl::set extent = getArrayExtent(Array, S);
  if (extent.is_empty()) {
    LLVM_DEBUG(dbgs() << "Cannot statically compute the array bounds!\n");
    assert(0 && "Cannot statically compute array bounds!");
    return std::make_tuple(negone, negone, negone, negone);
  }

  std::pair<isl::val, isl::val> dims_i = getDimensionBounds(ctx, extent, 0);
  std::pair<isl::val, isl::val> dims_j = getDimensionBounds(ctx, extent, 1);

  if (dims_i.first.eq(negone) || dims_i.second.eq(negone) ||
      dims_j.first.eq(negone) || dims_i.second.eq(negone)) {
    LLVM_DEBUG(dbgs() << "Cannot statically compute array bounds!\n");
    assert(0 && "Cannot statically compute array bounds!");
    return std::make_tuple(negone, negone, negone, negone);
  }

  return std::make_tuple(dims_i.first, dims_i.second, dims_j.first,
                         dims_j.second);
}

static int getBytesForArray(const ScopArrayInfo *sai, const Scop &s) {

  assert(sai->getNumberOfDimensions() == 2);

  auto bounds = getArrayBounds(sai, s);

  isl::val dim_i = std::get<1>(bounds).sub(std::get<0>(bounds)); 
  isl::val dim_j = std::get<3>(bounds).sub(std::get<2>(bounds));
  dim_i = dim_i.mul(dim_j);

  int total_elements = std::stoi(dim_i.to_str());  
  int size_element = sai->getElemSizeInBytes();
  return total_elements * size_element;

}

static std::pair<isl::val, isl::val> getVectorBounds(const ScopArrayInfo *sai, 
const Scop &s) {

  isl::set extent = getArrayExtent(sai, s);
  isl::ctx ctx = sai->getScop().getIslCtx();
  isl::val negone = isl::val::negone(ctx);

  if (extent.is_empty()) {
    assert(0 && "Cannot statically compute the array bounds!");
    return std::make_pair(negone, negone);
  }

  auto dims_i = getDimensionBounds(ctx, extent, 0);
  if (dims_i.first.eq(negone) || dims_i.second.eq(negone)) {
    assert(0 && "Cannot statically compute array bounds!");
    return dims_i;
  }

  return dims_i;
}

static int getBytesForVector(const ScopArrayInfo *sai, const Scop &s) {

  assert(sai->getNumberOfDimensions() == 1);

  auto bounds = getVectorBounds(sai, s);
  isl::val dim_i = bounds.second.sub(bounds.first);
  
  int total_elements = std::stoi(dim_i.to_str());
  int size_element = sai->getElemSizeInBytes();
  return total_elements * size_element;

} 

template <typename T, typename... Args>
static int computeSharedMemorySize(const Scop &s, const T arg, const Args... args) {

  static_assert(are_same<MemoryAccess*, Args...>{}, "must be of type MemoryAccess*");

  std::vector<MemoryAccess*> mem_accesses{};
  mem_accesses.push_back(arg);

  do_for([&](MemoryAccess * arg) {
    mem_accesses.push_back(arg);
  }, args...);

  int bytes = 0;

  for (const auto &mem_acc : mem_accesses) {
    ScopArrayInfo *sai =
      const_cast<ScopArrayInfo*>(mem_acc->getLatestScopArrayInfo());
    int dims = sai->getNumberOfDimensions();
    switch (dims) {
      case 1:
        bytes+= getBytesForVector(sai,s); break;
      case 2:
        bytes += getBytesForArray(sai,s); break;
      default: assert(0);
    }
  }

  return bytes;
}

// This will likely change in the near future.
// I really do not like to have this static variable
// pGemm. Another approach would be to allocate the 
// MMI structs with new and then walk the tree to collect
// the info you want.
static int computeSharedMemorySizeForGemm(const Scop &s) {

  int bytes = 0;
  for (auto const &MMI : pGemm.patternTys) {
    bytes += computeSharedMemorySize(s, MMI.A, MMI.B, MMI.WriteToC);
  }
  return bytes;
}

static int computeSharedMemorySizeForGemv(const Scop &s) {

  int bytes = 0;
  for (auto const &MVI : pGemv.patternTys) {
    bytes += computeSharedMemorySize(s, MVI.A, MVI.X, MVI.WriteToY);
  }
  return bytes;
}

static isl::schedule handleCimInitAndTearDown(isl::schedule schedule,
const Scop &s, std::function<int(const Scop &s)> f) {

  isl::schedule_node root = schedule.get_root().child(0);
  root = addCimStartUp(root);
  int bytes = f(s);
  root = addCimAllocateSharedMemory(root, bytes);
  root = addCimTearDown(root);
  schedule = root.root().get_schedule();
  return schedule;
}

static isl::schedule optimizeScheduleWithMatchersLate(isl::schedule schedule,
                                                      const Scop &s,
                                                      const Tactic &tac) {

  schedule = isGemmLikeLate(schedule, s, tac);
  if (lookUpScheduleTree(schedule, "gemm")) {
    // insert cim init and tear_down only if  
    // we detect the gemm pattern. We also graft a subtree
    // used to allocate the shared memory needed by the CIM
    // device.
    std::function<int(const Scop &s)> 
      computeSharedMemorySizeForGemmF = computeSharedMemorySizeForGemm; 
    schedule = handleCimInitAndTearDown(schedule, s, computeSharedMemorySizeForGemmF);
    LLVM_DEBUG(dbgs() << "Matchers: GEMM pattern detected!\n");
  }

  schedule = isGemvLikeLate(schedule, s);
  if (lookUpScheduleTree(schedule, "gemv")) {
    std::function<int(const Scop &s)>
      computeSharedMemorySizeForGemvF = computeSharedMemorySizeForGemv;
    schedule = handleCimInitAndTearDown(schedule, s, computeSharedMemorySizeForGemvF);
    LLVM_DEBUG(dbgs() << "Matchers: GEMV pattern detected!\n");
  }

  return schedule;
}

/// Simplify the schedule tree.
/// given a tree that looks like
///
/// schedule (i)
///    schedule (j)
///      anyTree
///
/// this will get simplify as
///
/// schedule(i,j)
///   anyTree
///
/// @param schedule_node: Current schedule node to be simplified.
isl::schedule_node simplifyTree(isl::schedule_node root) {

  isl::schedule_node parent, child, grandchild;
  auto matcher = [&]() {
    using namespace matchers;
    // clang-format off
    return band(parent,
      band(child,
        anyTree(grandchild)));
    //clang-format on
  }();
    
  auto merger = builders::ScheduleNodeBuilder();
  {
    using namespace builders;
    // clang-format off
    auto computeSched = [&]() {
      isl::multi_union_pw_aff sched =
        parent.band_get_partial_schedule().flat_range_product(
          child.band_get_partial_schedule());
      return sched;
    };
    // clang-format on
    auto st = [&]() { return subtreeBuilder(grandchild); };
    merger = band(computeSched, subtree(st));
  }

  root = replaceDFSPreorderRepeatedly(root, matcher, merger);
  return root.root();
}

static isl::schedule isGemmLikeEarly(isl::schedule schedule, Scop &s) {

  isl::schedule_node root = schedule.get_root();
  root = simplifyTree(root);

  // see ScheduleOptimizer.h
  pGemm.flush();

  // check gemm conditions:
  // 1. We mast have a single-map schedule
  // 2. The input dimension for the schedule must be >= 3
  auto hasGemmConditions = [&](isl::schedule_node band) {
    isl::union_map sched =
        isl::union_map::from(band.band_get_partial_schedule());
    if (sched.n_map() != 1) {
      LLVM_DEBUG(dbgs() << "hasGemmConditions: false due to n_map != 1\n");
      LLVM_DEBUG(dbgs() << "number of map is: " << sched.n_map() << "\n");
      return false;
    }
    isl::map schedAsMap = isl::map::from_union_map(sched);
    if (schedAsMap.dim(isl::dim::in) < 3) {
      LLVM_DEBUG(dbgs() << "hasGemmConditions: false due to in dim < 3\n");
      LLVM_DEBUG(dbgs() << "input dimension is :"
                        << schedAsMap.dim(isl::dim::in) << "\n");
      return false;
    }
    LLVM_DEBUG(dbgs() << "hasGemmConditions: true\n");
    return true;
  };

  // check gemm access pattern.
  auto containsMatrMul = [&](isl::schedule_node band) {
    MatMulInfoTyExtended MMI;

    isl::map scheduleAsMap = isl::map::from_union_map(
        isl::union_map::from(band.band_get_partial_schedule()));

    if (!checkAccessGemmStmt(s, scheduleAsMap, MMI)) {
      LLVM_DEBUG(
          dbgs() << "containsMatrMul: false due to checkAccessGemmStmt\n");
      return false;
    }
    pGemm.detected++;
    pGemm.patternTys.push_back(MMI);
    LLVM_DEBUG(dbgs() << "containsMatrMul: true\n");
    return true;
  };

  // This callback avoids entering an infinite loop
  // during recursion (wrapPatternDFSPreorder).
  // Specifically, the callback checks if the matcher
  // already fired.
  auto hasNotFired = [&](isl::schedule_node band) {
    if (!band.has_parent())
      return true;
    auto maybeMark = band.parent();
    if (isl_schedule_node_get_type(maybeMark.get()) != isl_schedule_node_mark)
      return true;
    auto markId = maybeMark.mark_get_id().to_str();
    markId = markId.substr(0, markId.find("@"));
    if ((markId.compare("gemm") == 0) || (markId.compare("gemm_init") == 0))
      return false;
    else
      return true;
  };

  // look for the gemm pattern
  auto matcherGemm = [&]() {
    using namespace matchers;
    // clang-format off
    return
      band(_and(hasNotFired, hasGemmConditions, containsMatrMul),
        leaf());
    // clang-format on
  }();

  root = wrapPatternDFSPreorder(s.getIslCtx(), &pGemm, "gemm", root.child(0),
                                matcherGemm);

  // early exit if we did not detect any core gemm stmt.
  // if we did detect a gemm pattern we also look for
  // a possible initialization stmt.
  if (!lookUpScheduleTree(root.root().get_schedule(), "gemm")) {
    return root.root().get_schedule();
  }

  // check conditions for init stmt.
  // 1. single-map schedule after removed redundant schedules.
  auto hasGemmInitConditions = [&](isl::schedule_node band) {
    isl::union_map sched =
        isl::union_map::from(band.band_get_partial_schedule());
    sched = removeAdditionalSchedules(sched, 2);
    if (sched.n_map() != 1) {
      LLVM_DEBUG(dbgs() << "hasGemmInitConditions: false due to n_map != 1\n");
      return false;
    }
    LLVM_DEBUG(dbgs() << "hasGemmInitConditions: true\n");
    return true;
  };

  // check the access pattern for the init stmt that must
  // be compliant with the gemm one.
  auto containsMatrMulInit = [&](isl::schedule_node band) {
    isl::union_map sched =
        isl::union_map::from(band.band_get_partial_schedule());
    sched = removeAdditionalSchedules(sched, 2);
    isl::map scheduleAsMap = isl::map::from_union_map(sched);
    if (!checkAccessGemmInitStmt(s, scheduleAsMap,
                                 pGemm.patternTys[pGemm.current])) {
      LLVM_DEBUG(
          dbgs()
          << "containsMatrMulInit: false due to checkAccessGemmInitStmt\n");
      return false;
    }
    LLVM_DEBUG(dbgs() << "containsMatrMulInit: true\n");
    pGemm.current++;
    return true;
  };

  // check if the init stmt dominates and is in the same
  // subtree of the gemm pattern.
  // NOTE: we may lose some potential optimization. We may
  // not find the init stmt if this rule does not apply.
  auto hasAnnotation = [&](isl::schedule_node mark) {
    isl::id markId = mark.mark_get_id();
    std::string idAsString = markId.to_str();
    idAsString = idAsString.substr(0, idAsString.find("@"));
    if (idAsString.compare("gemm") == 0) {
      LLVM_DEBUG(dbgs() << "hasAnnotation -> true\n");
      return true;
    }
    LLVM_DEBUG(dbgs() << "hasAnnotation -> false\n");
    return false;
  };

  isl::schedule_node subtree;
  isl::schedule_node dummySubtree;
  auto matcherInit = [&]() {
    using namespace matchers;
    // clang-format off
    return
      band(_and(hasChild(mark(hasAnnotation, band(anyTree(dummySubtree)))), 
                hasNotFired,
                hasGemmInitConditions, 
                containsMatrMulInit),
        anyTree(subtree));
    // clang-format on
  }();

  root = wrapPatternDFSPreorder(s.getIslCtx(), &pGemm, "gemm_init", root,
                                matcherInit);
  pGemm.current = 0;
  return root.root().get_schedule();
}

static isl::schedule optimizeScheduleWithMatchersEarly(isl::schedule schedule,
                                                       Scop &s) {

  assert(0 &&
      "Use optimizeScheduleWithMatchersLate, this flow is not available atm");
  schedule = isGemmLikeEarly(schedule, s);
  if (lookUpScheduleTree(schedule, "gemm")) {
    LLVM_DEBUG(dbgs() << "Matchers: GEMM pattern detected!\n");
  }
  return schedule;
}

bool IslScheduleOptimizer::runOnScop(Scop &S) {
  // Skip SCoPs in case they're already optimised by PPCGCodeGeneration
  if (S.isToBeSkipped())
    return false;

  // Skip empty SCoPs but still allow code generation as it will delete the
  // loops present but not needed.
  if (S.getSize() == 0) {
    S.markAsOptimized();
    return false;
  }

  const Dependences &D =
      getAnalysis<DependenceInfo>().getDependences(Dependences::AL_Statement);

  if (D.getSharedIslCtx() != S.getSharedIslCtx()) {
    LLVM_DEBUG(dbgs() << "DependenceInfo for another SCoP/isl_ctx\n");
    return false;
  }

  if (!D.hasValidDependences())
    return false;

  isl_schedule_free(LastSchedule);
  LastSchedule = nullptr;

  // Build input data.
  int ValidityKinds =
      Dependences::TYPE_RAW | Dependences::TYPE_WAR | Dependences::TYPE_WAW;
  int ProximityKinds;

  if (OptimizeDeps == "all")
    ProximityKinds =
        Dependences::TYPE_RAW | Dependences::TYPE_WAR | Dependences::TYPE_WAW;
  else if (OptimizeDeps == "raw")
    ProximityKinds = Dependences::TYPE_RAW;
  else {
    errs() << "Do not know how to optimize for '" << OptimizeDeps << "'"
           << " Falling back to optimizing all dependences.\n";
    ProximityKinds =
        Dependences::TYPE_RAW | Dependences::TYPE_WAR | Dependences::TYPE_WAW;
  }

  isl::union_set Domain = S.getDomains();

  if (!Domain)
    return false;

  ScopsProcessed++;
  walkScheduleTreeForStatistics(S.getScheduleTree(), 0);

  isl::union_map Validity = D.getDependences(ValidityKinds);
  isl::union_map Proximity = D.getDependences(ProximityKinds);

  // Simplify the dependences by removing the constraints introduced by the
  // domains. This can speed up the scheduling time significantly, as large
  // constant coefficients will be removed from the dependences. The
  // introduction of some additional dependences reduces the possible
  // transformations, but in most cases, such transformation do not seem to be
  // interesting anyway. In some cases this option may stop the scheduler to
  // find any schedule.
  if (SimplifyDeps == "yes") {
    Validity = Validity.gist_domain(Domain);
    Validity = Validity.gist_range(Domain);
    Proximity = Proximity.gist_domain(Domain);
    Proximity = Proximity.gist_range(Domain);
  } else if (SimplifyDeps != "no") {
    errs() << "warning: Option -polly-opt-simplify-deps should either be 'yes' "
              "or 'no'. Falling back to default: 'yes'\n";
  }

  LLVM_DEBUG(dbgs() << "\n\nCompute schedule from: ");
  LLVM_DEBUG(dbgs() << "Domain := " << Domain << ";\n");
  LLVM_DEBUG(dbgs() << "Proximity := " << Proximity << ";\n");
  LLVM_DEBUG(dbgs() << "Validity := " << Validity << ";\n");

  unsigned IslSerializeSCCs;

  if (FusionStrategy == "max") {
    IslSerializeSCCs = 0;
  } else if (FusionStrategy == "min") {
    IslSerializeSCCs = 1;
  } else {
    errs() << "warning: Unknown fusion strategy. Falling back to maximal "
              "fusion.\n";
    IslSerializeSCCs = 0;
  }

  int IslMaximizeBands;

  if (MaximizeBandDepth == "yes") {
    IslMaximizeBands = 1;
  } else if (MaximizeBandDepth == "no") {
    IslMaximizeBands = 0;
  } else {
    errs() << "warning: Option -polly-opt-maximize-bands should either be 'yes'"
              " or 'no'. Falling back to default: 'yes'\n";
    IslMaximizeBands = 1;
  }

  int IslOuterCoincidence;

  if (OuterCoincidence == "yes") {
    IslOuterCoincidence = 1;
  } else if (OuterCoincidence == "no") {
    IslOuterCoincidence = 0;
  } else {
    errs() << "warning: Option -polly-opt-outer-coincidence should either be "
              "'yes' or 'no'. Falling back to default: 'no'\n";
    IslOuterCoincidence = 0;
  }

  isl_ctx *Ctx = S.getIslCtx().get();

  isl_options_set_schedule_outer_coincidence(Ctx, IslOuterCoincidence);
  isl_options_set_schedule_serialize_sccs(Ctx, IslSerializeSCCs);
  isl_options_set_schedule_maximize_band_depth(Ctx, IslMaximizeBands);
  isl_options_set_schedule_max_constant_term(Ctx, MaxConstantTerm);
  isl_options_set_schedule_max_coefficient(Ctx, MaxCoefficient);
  isl_options_set_tile_scale_tile_loops(Ctx, 0);

  auto OnErrorStatus = isl_options_get_on_error(Ctx);
  isl_options_set_on_error(Ctx, ISL_ON_ERROR_CONTINUE);

  auto SC = isl::schedule_constraints::on_domain(Domain);
  SC = SC.set_proximity(Proximity);
  SC = SC.set_validity(Validity);
  SC = SC.set_coincidence(Validity);
  auto Schedule = SC.compute_schedule();
  isl_options_set_on_error(Ctx, OnErrorStatus);

  walkScheduleTreeForStatistics(Schedule, 1);

  // In cases the scheduler is not able to optimize the code, we just do not
  // touch the schedule.
  if (!Schedule)
    return false;

  ScopsRescheduled++;

  LLVM_DEBUG({
    auto *P = isl_printer_to_str(Ctx);
    P = isl_printer_set_yaml_style(P, ISL_YAML_STYLE_BLOCK);
    P = isl_printer_print_schedule(P, Schedule.get());
    auto *str = isl_printer_get_str(P);
    dbgs() << "NewScheduleTree: \n" << str << "\n";
    free(str);
    isl_printer_free(P);
  });

  isl::schedule NewSchedule;

  Tactic tac = Tactic::TILING;

  if (MatcherOptLate) {
    NewSchedule = optimizeScheduleWithMatchersLate(Schedule, S, tac);
  }
  if (MatcherOptEarly) {
    NewSchedule = optimizeScheduleWithMatchersEarly(S.getScheduleTree(), S);
  }
  if (!MatcherOptLate && !MatcherOptEarly) {
    Function &F = S.getFunction();
    auto *TTI = &getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
    const OptimizerAdditionalInfoTy OAI = {TTI, const_cast<Dependences *>(&D)};
    NewSchedule = ScheduleTreeOptimizer::optimizeSchedule(Schedule, &OAI);
  }
  walkScheduleTreeForStatistics(NewSchedule, 2);

  // if (!ScheduleTreeOptimizer::isProfitableSchedule(S, NewSchedule))
  //  return false;

  auto ScopStats = S.getStatistics();
  ScopsOptimized++;
  NumAffineLoopsOptimized += ScopStats.NumAffineLoops;
  NumBoxedLoopsOptimized += ScopStats.NumBoxedLoops;

  S.setScheduleTree(NewSchedule);
  S.markAsOptimized();

  LLVM_DEBUG(dbgs() << "********************************\n");
  LLVM_DEBUG(dbgs() << "TO CODEGEN\n");
  LLVM_DEBUG(dbgs() << NewSchedule.get_root().to_str() << "\n");
  LLVM_DEBUG(dbgs() << "********************************\n");

  if (OptimizedScops)
    errs() << S;

  return false;
}

void IslScheduleOptimizer::printScop(raw_ostream &OS, Scop &) const {
  isl_printer *p;
  char *ScheduleStr;

  OS << "Calculated schedule:\n";

  if (!LastSchedule) {
    OS << "n/a\n";
    return;
  }

  p = isl_printer_to_str(isl_schedule_get_ctx(LastSchedule));
  p = isl_printer_print_schedule(p, LastSchedule);
  ScheduleStr = isl_printer_get_str(p);
  isl_printer_free(p);

  OS << ScheduleStr << "\n";
}

void IslScheduleOptimizer::getAnalysisUsage(AnalysisUsage &AU) const {
  ScopPass::getAnalysisUsage(AU);
  AU.addRequired<DependenceInfo>();
  AU.addRequired<TargetTransformInfoWrapperPass>();

  AU.addPreserved<DependenceInfo>();
}

Pass *polly::createIslScheduleOptimizerPass() {
  return new IslScheduleOptimizer();
}

INITIALIZE_PASS_BEGIN(IslScheduleOptimizer, "polly-opt-isl",
                      "Polly - Optimize schedule of SCoP", false, false);
INITIALIZE_PASS_DEPENDENCY(DependenceInfo);
INITIALIZE_PASS_DEPENDENCY(ScopInfoRegionPass);
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass);
INITIALIZE_PASS_END(IslScheduleOptimizer, "polly-opt-isl",
                    "Polly - Optimize schedule of SCoP", false, false)
