"""
5 Diverse Coding Prompts for CIMBA Validation
Each prompt exercises different language patterns and attention structures.
"""

PROMPTS = [
    # 1. Rust: Binary Search - precise control flow, lifetimes
    """Implement a binary search function in Rust that finds the index of a target value in a sorted slice. Handle the case where the target is not found by returning None. Use proper Rust idioms with Option types.

```rust
fn binary_search<T: Ord>(slice: &[T], target: &T) -> Option<usize> {
    let mut left = 0;
    let mut right = slice.len();
    
    while left < right {
        let mid = left + (right - left) / 2;
        match slice[mid].cmp(target) {
            std::cmp::Ordering::Equal => return Some(mid),
            std::cmp::Ordering::Less => left = mid + 1,
            std::cmp::Ordering::Greater => right = mid,
        }
    }
    None
}
```""",

    # 2. Python: Decorator Pattern - nested closures, higher-order functions
    """Write a Python decorator that implements retry logic with exponential backoff. The decorator should accept parameters for max retries and base delay.

```python
import functools
import time
from typing import Callable, TypeVar

T = TypeVar('T')

def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    delay = base_delay * (2 ** attempt)
                    time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator
```""",

    # 3. SQL: Window Functions - complex aggregation, partitioning
    """Write a SQL query that calculates running totals and ranks employees by salary within each department, showing the percentage each salary represents of the department total.

```sql
SELECT 
    employee_id,
    department_id,
    salary,
    SUM(salary) OVER (
        PARTITION BY department_id 
        ORDER BY hire_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS running_total,
    RANK() OVER (
        PARTITION BY department_id 
        ORDER BY salary DESC
    ) AS salary_rank,
    ROUND(
        100.0 * salary / SUM(salary) OVER (PARTITION BY department_id),
        2
    ) AS pct_of_dept_total
FROM employees
ORDER BY department_id, salary DESC;
```""",

    # 4. C++: RAII Template - resource management, move semantics
    """Implement a C++ RAII wrapper template that manages any resource with custom deleter support, similar to unique_ptr but with logging capabilities.

```cpp
#include <functional>
#include <iostream>
#include <utility>

template<typename T, typename Deleter = std::default_delete<T>>
class ResourceGuard {
private:
    T* resource_;
    Deleter deleter_;
    bool owns_;

public:
    explicit ResourceGuard(T* res = nullptr, Deleter d = Deleter())
        : resource_(res), deleter_(std::move(d)), owns_(res != nullptr) {
        if (owns_) std::cout << "Acquired resource\\n";
    }

    ~ResourceGuard() {
        if (owns_ && resource_) {
            std::cout << "Releasing resource\\n";
            deleter_(resource_);
        }
    }

    ResourceGuard(ResourceGuard&& other) noexcept
        : resource_(other.resource_), deleter_(std::move(other.deleter_)), owns_(other.owns_) {
        other.owns_ = false;
        other.resource_ = nullptr;
    }

    ResourceGuard& operator=(ResourceGuard&& other) noexcept {
        if (this != &other) {
            if (owns_ && resource_) deleter_(resource_);
            resource_ = other.resource_;
            deleter_ = std::move(other.deleter_);
            owns_ = other.owns_;
            other.owns_ = false;
            other.resource_ = nullptr;
        }
        return *this;
    }

    ResourceGuard(const ResourceGuard&) = delete;
    ResourceGuard& operator=(const ResourceGuard&) = delete;

    T* get() const { return resource_; }
    T& operator*() const { return *resource_; }
    T* operator->() const { return resource_; }
    explicit operator bool() const { return owns_ && resource_; }
};
```""",

    # 5. Technical Explanation: Tail Recursion - conceptual + code
    """Explain tail recursion and demonstrate how to convert a standard recursive factorial function to tail-recursive form. Show both versions.

Tail recursion is an optimization technique where the recursive call is the final operation in a function. This allows the compiler to reuse the current stack frame instead of creating a new one, converting recursion into iteration.

Standard recursive factorial (not tail-recursive):
```python
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)  # Multiplication happens AFTER recursive call
```

Tail-recursive factorial using an accumulator:
```python
def factorial_tail(n, accumulator=1):
    if n <= 1:
        return accumulator
    return factorial_tail(n - 1, n * accumulator)  # No operation after call
```

The key insight is that the accumulator carries the intermediate result, so when the base case is reached, the final answer is immediately available without unwinding the stack. Languages like Scheme, Scala, and some C++ compilers optimize tail calls automatically.""",
]

# Short identifiers for logging
PROMPT_NAMES = [
    "rust_binary_search",
    "python_decorator",
    "sql_window_functions",
    "cpp_raii_template",
    "tail_recursion_explanation",
]
