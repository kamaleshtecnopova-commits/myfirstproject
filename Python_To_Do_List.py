# To-Do List Application

tasks = []

def add_task():
    description = input("Enter task description: ")
    tasks.append({"desc": description, "completed": False})
    print("Task added.\n")

def view_tasks():
    if not tasks:
        print("No tasks available.\n")
        return
    for i, task in enumerate(tasks, 1):
        status = "[COMPLETED]" if task["completed"] else ""
        print(f"{i}. {task['desc']} {status}")
    print()

def mark_complete():
    try:
        num = int(input("Enter task number to mark complete: "))
        if 1 <= num <= len(tasks):
            tasks[num - 1]["completed"] = True
            print(f"Task {num} marked as complete.\n")
        else:
            print("Invalid task number.\n")
    except ValueError:
        print("Please enter a valid number.\n")

def delete_task():
    try:
        num = int(input("Enter task number to delete: "))
        if 1 <= num <= len(tasks):
            del tasks[num - 1]
            print(f"Task {num} deleted.\n")
        else:
            print("Invalid task number.\n")
    except ValueError:
        print("Please enter a valid number.\n")

def main():
    while True:
        print("To-Do List Application")
        print("---------------------")
        print("1. Add Task")
        print("2. View Tasks")
        print("3. Mark Task Complete")
        print("4. Delete Task")
        print("5. Exit\n")
        choice = input("Enter your choice: ")

        if choice == "1":
            add_task()
        elif choice == "2":
            view_tasks()
        elif choice == "3":
            mark_complete()
        elif choice == "4":
            delete_task()
        elif choice == "5":
            print("Exiting To-Do List.")
            break
        else:
            print("Invalid choice. Please try again.\n")

if __name__ == "__main__":
    main()
