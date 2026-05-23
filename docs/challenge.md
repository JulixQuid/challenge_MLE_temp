### Instructions

1. Create a repository in **github** and copy all the challenge content into it. Remember that the repository must be **public**. DONE

2. Use the **main** branch for any official release that we should review. It is highly recommended to use [GitFlow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow) development practices. **NOTE: do not delete your development branches.** DONE
   
3. Please, do not change the structure of the challenge (names of folders and files). DONE
   
4. All the documentation and explanations that you have to give us must go in the `challenge.md` file inside `docs` folder. DONE

5. To send your challenge, you must do a `POST` request to:
    `https://advana-challenge-check-api-cr-k4hdbggvoq-uc.a.run.app/software-engineer`
    This is an example of the `body` you must send:
    ```json
    {
      "name": "Juan Perez",
      "mail": "juan.perez@example.com",
      "github_url": "https://github.com/juanperez/latam-challenge.git",
      "api_url": "https://juan-perez.api"
    }
    ```
    ##### ***PLEASE, SEND THE REQUEST ONCE.***

    If your request was successful, you will receive this message:
    ```json
    {
      "status": "OK",
      "detail": "your request was received"
    }
    ```


***NOTE: We recommend to send the challenge even if you didn't manage to finish all the parts.***

### Context:

We need to operationalize the data science work for the airport team. For this, we have decided to enable an `API` in which they can consult the delay prediction of a flight.

*We recommend reading the entire challenge (all its parts) before you start developing.*

### Part I

In order to operationalize the model, transcribe the `.ipynb` file into the `model.py` file:

- If you find any bug, fix it.
- The DS proposed a few models in the end. Choose the best model at your discretion, argue why. **It is not necessary to make improvements to the model.**
- Apply all the good programming practices that you consider necessary in this item.
- The model should pass the tests by running `make model-test`. (Done)

> **Note:**
> - **You cannot** remove or change the name or arguments of **provided** methods.
> - **You can** change/complete the implementation of the provided methods.
> - **You can** create the extra classes and methods you deem necessary.

### Part II

Deploy the model in an `API` with `FastAPI` using the `api.py` file.

- The `API` should pass the tests by running `make api-test`.

> **Note:** 
> - **You cannot** use other framework.

### Part III

Deploy the `API` in your favorite cloud provider (we recomend to use GCP).

- Put the `API`'s url in the `Makefile` (`line 26`).
- The `API` should pass the tests by running `make stress-test`.

> **Note:** 
> - **It is important that the API is deployed until we review the tests.**

### Part IV

We are looking for a proper `CI/CD` implementation for this development.

- Create a new folder called `.github` and copy the `workflows` folder that we provided inside it.
- Complete both `ci.yml` and `cd.yml`(consider what you did in the previous parts).