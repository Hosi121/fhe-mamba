#pragma once

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <functional>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace fhemamba {

// CPU producers, one consumer, ordered and bounded. No work is submitted
// after cancellation. Destruction joins even when a consumer or producer throws.
// Captured input must outlive the queue. Producers must not call GPU APIs;
// with multiple workers, their shared captured inputs must be thread-safe.
// The admission window counts both in-flight and ready items. Completion order
// may differ, but consumption order and the payload bound do not change.
template <class Item>
class BoundedPrefetch {
 public:
  template <class Produce>
  BoundedPrefetch(std::size_t count, std::size_t capacity, Produce produce,
                  std::size_t workers = 1)
      : capacity_(capacity) {
    if (!capacity_) throw std::invalid_argument("prefetch capacity must be positive");
    if (!workers) throw std::invalid_argument("prefetch workers must be positive");
    if (workers > 1) {
      if constexpr (!std::is_copy_constructible_v<Produce>) {
        throw std::invalid_argument("parallel prefetch requires a copyable producer");
      } else {
        parallel_count_ = count;
        slots_.resize(capacity_);
        const auto active = std::min({workers, capacity_, count});
        workers_.reserve(active);
        try {
          for (std::size_t worker = 0; worker < active; ++worker) {
            workers_.emplace_back([this, produce, worker]() mutable {
              try {
                while (true) {
                  std::size_t index;
                  {
                    std::unique_lock lock(mutex_);
                    ready_.wait(lock, [&] {
                      return cancelled_ || assigned_ == parallel_count_ ||
                             assigned_ - consumed_ < capacity_;
                    });
                    if (cancelled_ || assigned_ == parallel_count_) return;
                    index = assigned_++;
                  }
                  auto item = produce(index);
                  {
                    std::lock_guard lock(mutex_);
                    if (cancelled_) return;
                    slots_[index % capacity_].emplace(std::move(item));
                    peak_size_ = std::max(peak_size_, ++ready_count_);
                  }
                  ready_.notify_all();
                }
              } catch (...) {
                {
                  std::lock_guard lock(mutex_);
                  if (!error_) error_ = std::current_exception();
                  cancelled_ = true;
                }
                ready_.notify_all();
              }
            });
          }
        } catch (...) {
          cancel_and_join();
          throw;
        }
        return;
      }
    }
    worker_ = std::thread([this, count, produce = std::move(produce)]() mutable {
      try {
        for (std::size_t i = 0; i < count; ++i) {
          {
            std::unique_lock lock(mutex_);
            ready_.wait(lock, [&] { return cancelled_ || queue_.size() < capacity_; });
            if (cancelled_) return;
          }
          auto item = produce(i);
          {
            std::lock_guard lock(mutex_);
            if (cancelled_) return;
            queue_.push_back(std::move(item));
            peak_size_ = std::max(peak_size_, queue_.size());
          }
          ready_.notify_all();
        }
        { std::lock_guard lock(mutex_); done_ = true; }
      } catch (...) {
        std::lock_guard lock(mutex_);
        error_ = std::current_exception();
      }
      ready_.notify_all();
    });
  }
  BoundedPrefetch(const BoundedPrefetch&) = delete;
  auto operator=(const BoundedPrefetch&) -> BoundedPrefetch& = delete;
  ~BoundedPrefetch() { cancel_and_join(); }
  auto pop() -> Item {
    std::unique_lock lock(mutex_);
    if (!slots_.empty()) {
      ready_.wait(lock, [&] {
        return error_ || consumed_ == parallel_count_ || slots_[consumed_ % capacity_].has_value();
      });
      if (error_) std::rethrow_exception(error_);
      if (consumed_ == parallel_count_) throw std::out_of_range("prefetch exhausted");
      auto& slot = slots_[consumed_ % capacity_];
      auto item = std::move(*slot);
      slot.reset();
      ++consumed_; --ready_count_;
      lock.unlock();
      ready_.notify_all();
      return item;
    }
    ready_.wait(lock, [&] { return error_ || done_ || !queue_.empty(); });
    if (error_) std::rethrow_exception(error_);
    if (queue_.empty()) throw std::out_of_range("prefetch exhausted");
    auto item = std::move(queue_.front());
    queue_.pop_front();
    lock.unlock();
    ready_.notify_all();
    return item;
  }
  auto peak_size() const -> std::size_t {
    std::lock_guard lock(mutex_);
    return peak_size_;
  }

 private:
  void cancel_and_join() {
    { std::lock_guard lock(mutex_); cancelled_ = true; }
    ready_.notify_all();
    if (worker_.joinable()) worker_.join();
    for (auto& worker : workers_) worker.join();
  }
  std::size_t capacity_, peak_size_ = 0;
  mutable std::mutex mutex_;
  std::condition_variable ready_;
  std::deque<Item> queue_;
  bool cancelled_ = false, done_ = false;
  std::exception_ptr error_;
  std::thread worker_;
  std::vector<std::thread> workers_;
  std::vector<std::optional<Item>> slots_;
  std::size_t assigned_ = 0, consumed_ = 0, parallel_count_ = 0, ready_count_ = 0;
};

}  // namespace fhemamba
