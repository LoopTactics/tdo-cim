#include "polly/Matchers.h"
#include "polly/Die.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <stack>

using namespace llvm;

#define DEBUG_TYPE "polly-matcher"

namespace matchers {

thread_local std::vector<isl::schedule_node>
    ScheduleNodeMatcher::dummyMultiCaptureData_;

bool ScheduleNodeMatcher::isMatching(const ScheduleNodeMatcher &matcher,
                                     isl::schedule_node node) {

  if (!node.get()) {
    return false;
  }

  if (matcher.current_ == ScheduleNodeType::AnyTree) {
    matcher.capture_ = node;
    return true;
  }

  // FilterForest implementation
  // It allows to skip a group of filter nodes, that
  // have leaf as their only child.
  //
  // sequence
  //  filter
  //    leaf
  //  filter
  //    leaf
  //  ...
  //  filter
  //    schedule
  //      leaf
  //

  if (matcher.current_ == ScheduleNodeType::Sequence) {

    if (isl_schedule_node_get_type(node.get()) != isl_schedule_node_sequence)
      return false;

    // is the next matcher filterForest?
    bool nextIsFilterForest =
        matcher.children_.at(0).current_ == ScheduleNodeType::FilterForest;

    if (nextIsFilterForest) {

      node = node.child(0);
      assert(isl_schedule_node_get_type(node.get()) ==
                 isl_schedule_node_filter &&
             "expect a filter node");

      // exit if we are not dealing with a filter like:
      // filter
      //  leaf
      if (isl_schedule_node_get_type(node.child(0).get()) !=
          isl_schedule_node_leaf)
        return false;

      assert(matcher.children_.at(0).current_ ==
                 ScheduleNodeType::FilterForest &&
             "expect filter forest as child.");
      assert(matcher.children_.at(0).needToCapture_ == true &&
             "expect it to be true");

      matcher.children_.at(0).multiCapture_.push_back(node);

      while (isl_schedule_node_has_next_sibling(node.get()) &&
             isl_schedule_node_get_type(node.next_sibling().get()) ==
                 isl_schedule_node_filter &&
             isl_schedule_node_get_type(node.next_sibling().child(0).get()) ==
                 isl_schedule_node_leaf) {
        node = node.next_sibling();
        matcher.children_.at(0).multiCapture_.push_back(node);
      }

      // if filter forest is the terminating node matcher
      // capture also the last filter.
      if (!isl_schedule_node_has_next_sibling(node.get()) &&
          matcher.children_.size() == 1) {
        matcher.children_.at(0).multiCapture_.push_back(node);
        return true;
      }

      // expect a sibling node filter after filterForest,
      // if filterForest is non terminating.
      // In this case the current matcher is the sequence.
      // while the current node is the last node that looks like
      // filter
      //  leaf
      if (!isl_schedule_node_has_next_sibling(node.get()) &&
          matcher.children_.size() >= 1)
        return false;

      // check node after group of filter + leaf.
      if (!isMatching(matcher.children_.at(1), node.next_sibling()))
        return false;
      else
        return true;
    } // end if nextIsFilterForest
  }

  // Below, we traverse children in order.  If AnyForest match is requested, it
  // matches this node and all its next siblings.  Currently, AnyForest cannot
  // be combined with other matchers, so we expect no previous sibling and
  // capture all next siblings.
  // The combination of AnyForest with other types would require shell
  // wildcard-like matching.
  // We do this in the start rather than in the child-visiting loop because
  // AnyForest can be the root of the matcher and cannot be converted to an isl
  // type below.
  if (matcher.current_ == ScheduleNodeType::AnyForest) {
    if (node.has_previous_sibling()) {
      ISLUTILS_DIE("AnyForest matcher combined with other types");
    }
    matcher.multiCapture_.clear();
    do {
      matcher.multiCapture_.push_back(node);
    } while (node.has_next_sibling() && (node = node.next_sibling()));
    return true;
  }

  if (toIslType(matcher.current_) != isl_schedule_node_get_type(node.get())) {
    return false;
  }

  if (matcher.nodeCallback_ && !matcher.nodeCallback_(node)) {
    return false;
  }

  // Check that the number of children matches unless the only matcher child is
  // AnyForest.  In the latter case, check that the tree node has at least one
  // child.
  size_t nChildren =
      static_cast<size_t>(isl_schedule_node_n_children(node.get()));
  bool nextIsAnyForest =
      matcher.children_.size() == 1 &&
      matcher.children_.at(0).current_ == ScheduleNodeType::AnyForest;
  if (matcher.children_.size() != nChildren && !nextIsAnyForest) {
    return false;
  }
  if (nextIsAnyForest && nChildren == 0) {
    return false;
  }

  for (size_t i = 0; i < nChildren; ++i) {
    if (!isMatching(matcher.children_.at(i), node.child(i))) {
      return false;
    }
    // Return after visiting the first child if we match AnyForest because it
    // included all the siblings.
    if (nextIsAnyForest) {
      return true;
    }
  }

  // avoid to capture the node, if the matcher
  // is not supposed to capture any node.
  if (matcher.needToCapture_) {
    matcher.capture_ = node;
  }

  return true;
}

static bool hasPreviousSiblingImpl(isl::schedule_node node,
                                   const ScheduleNodeMatcher &siblingMatcher) {
  while (isl_schedule_node_has_previous_sibling(node.get()) == isl_bool_true) {
    node = isl::manage(isl_schedule_node_previous_sibling(node.release()));
    if (ScheduleNodeMatcher::isMatching(siblingMatcher, node)) {
      return true;
    }
  }
  return false;
}

static bool hasNextSiblingImpl(isl::schedule_node node,
                               const ScheduleNodeMatcher &siblingMatcher) {
  while (isl_schedule_node_has_next_sibling(node.get()) == isl_bool_true) {
    node = isl::manage(isl_schedule_node_next_sibling(node.release()));
    if (ScheduleNodeMatcher::isMatching(siblingMatcher, node)) {
      return true;
    }
  }
  return false;
}

static bool
hasImmediateNextSiblingImpl(isl::schedule_node node,
                            const ScheduleNodeMatcher &siblingMatcher) {

  if (isl_schedule_node_has_next_sibling(node.get()) == isl_bool_true) {
    node = isl::manage(isl_schedule_node_next_sibling(node.release()));
    if (ScheduleNodeMatcher::isMatching(siblingMatcher, node)) {
      return true;
    }
  }
  return false;
}

std::function<bool(isl::schedule_node)>
hasImmediateNextSibling(const ScheduleNodeMatcher &siblingMatcher) {
  return std::bind(hasImmediateNextSiblingImpl, std::placeholders::_1,
                   siblingMatcher);
}

std::function<bool(isl::schedule_node)>
hasPreviousSibling(const ScheduleNodeMatcher &siblingMatcher) {
  return std::bind(hasPreviousSiblingImpl, std::placeholders::_1,
                   siblingMatcher);
}

std::function<bool(isl::schedule_node)>
hasNextSibling(const ScheduleNodeMatcher &siblingMatcher) {
  return std::bind(hasNextSiblingImpl, std::placeholders::_1, siblingMatcher);
}

std::function<bool(isl::schedule_node)>
hasSibling(const ScheduleNodeMatcher &siblingMatcher) {
  return [siblingMatcher](isl::schedule_node node) {
    return hasPreviousSiblingImpl(node, siblingMatcher) ||
           hasNextSiblingImpl(node, siblingMatcher);
  };
}

static bool hasChildImpl(isl::schedule_node node,
                         const ScheduleNodeMatcher &descendantMatcher) {

  std::stack<isl::schedule_node> nodeStack;
  nodeStack.push(node);

  while (nodeStack.empty() == false) {
    node = nodeStack.top();
    nodeStack.pop();

    if (ScheduleNodeMatcher::isMatching(descendantMatcher, node))
      return true;

    size_t n_children =
        static_cast<size_t>(isl_schedule_node_n_children(node.get()));
    for (size_t i = 0; i < n_children; i++)
      nodeStack.push(node.child(i));
  }

  return false;
}

std::function<bool(isl::schedule_node)>
hasChild(const ScheduleNodeMatcher &descendantMatcher) {
  return std::bind(hasChildImpl, std::placeholders::_1, descendantMatcher);
}

std::function<bool(isl::schedule_node)>
hasDescendant(const ScheduleNodeMatcher &descendantMatcher) {
  isl::schedule_node n;
  return [descendantMatcher](isl::schedule_node node) {
    // Cannot use capturing lambdas as C function pointers.
    struct Data {
      bool found;
      const ScheduleNodeMatcher &descendantMatcher;
    };
    Data data{false, descendantMatcher};

    auto r = isl_schedule_node_foreach_descendant_top_down(
        node.get(),
        [](__isl_keep isl_schedule_node *cn, void *user) -> isl_bool {
          auto data = static_cast<Data *>(user);
          if (data->found) {
            return isl_bool_false;
          }

          auto n = isl::manage_copy(cn);
          data->found =
              ScheduleNodeMatcher::isMatching(data->descendantMatcher, n);
          return data->found ? isl_bool_false : isl_bool_true;
        },
        &data);
    return r == isl_stat_ok && data.found;
  };
}

} // namespace matchers
