{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyN/d94yKShaGIbXldHHim8y",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/frivblon1712/EvaluacionIA/blob/main/evaluacionia1.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "eSZrE6j2kOSI"
      },
      "outputs": [],
      "source": [
        "import tensorflow as tf\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "from sklearn.model_selection import train_test_split\n",
        "from keras.models import Sequential\n",
        "from keras.layers import Dense"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Datos de ejemplo: desde bit hasta Yottabyte\n",
        "bytes_data = np.array([1024, 1048576, 1073741824], dtype=float)\n",
        "kilobytes_data = bytes_data / 1024"
      ],
      "metadata": {
        "id": "j68K4uClkiW9"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "capa = tf.keras.layers.Dense(units=1, input_shape=[1])\n",
        "modelo = tf.keras.Sequential([capa])"
      ],
      "metadata": {
        "id": "p0ivYWVC8mc0"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "modelo.compile(\n",
        "    optimizer = tf.keras.optimizers.Adam(0.1),\n",
        "    loss='mean_squared_error'\n",
        ")"
      ],
      "metadata": {
        "id": "NAjlE4T6kpHf"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(\"comenzando entrenamiento\")\n",
        "historial=modelo.fit(bytes_data, kilobytes_data, epochs=1000, verbose=False)\n",
        "print(\"modelo entrenado!!!\")"
      ],
      "metadata": {
        "id": "98-aymvikqm4",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "02b454a0-57a9-4df7-a511-0316271b477e"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "comenzando entrenamiento\n",
            "modelo entrenado!!!\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "plt.xlabel(\"# Epoca\")\n",
        "plt.ylabel(\"Magnitud de perdida\")\n",
        "plt.plot(historial.history[\"loss\"])"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 482
        },
        "id": "y1hvouqLksQR",
        "outputId": "83ec0e41-8c08-4c3e-dbdf-54cf93eec07c"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "[<matplotlib.lines.Line2D at 0x7ab320389f30>]"
            ]
          },
          "metadata": {},
          "execution_count": 7
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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RkaGJk6c6JWaf6t8IUSWAQIAwDpVCkAdO3aUzWaTYRjmOjbS2YG85x4rKyvzcIneN3bsWKWnp5vvHQ6HYmNjvfK7zs4CIwEBAGCVKgWgPXv2mD9v3rxZjz/+uJ544gklJydLOt019corr2jKlCneqfKMqKgo5efnuxzLz89XaGiogoOD5efnJz8/v0qviYqKOu99AwMDfTaFv7wFqIwmIAAALFOlABQfH2/+fM899+jVV1/VHXfcYR7r0KGDYmNjNW7cuAsOYr5cycnJWr58ucuxVatWmUEsICBAnTt3VmZmplmH0+lUZmamRo4c6bW63GG30wUGAIDV3J4Gv3XrViUkJFQ4npCQoG+//datexUVFSk7O1vZ2dmSTrc0ZWdnKycnR9LprqnBgweb1z/00EP68ccfNWbMGO3cuVN/+9vf9I9//EN/+tOfzGvS09P15ptvav78+dqxY4cefvhhFRcXKy0tzd2v6hV2VoIGAMBybs8Ca926tTIyMvTWW28pICBA0unZVxkZGWrdurVb99qwYYPL4onl43CGDBmiefPmKTc31wxD0umQ9dFHH+lPf/qTpk+friZNmuitt95SamqqeU3//v11+PBhjR8/Xnl5eerYsaNWrFhRYWC0Vcq7wBgDBACAddxeB2j9+vXq06ePDMMwZ3xt2bJFNptNH3zwgbp27eqVQn3Jm+sATV31nV7N/F733RCvSf3aefTeAADUZl5dB6hr16768ccf9e6775pTy/v3768//vGPCgkJubSKaxG6wAAAsN4lLYQYEhKiBx980NO11Apn1wEiAAEAYBW3B0Hj8viVzwJzWlwIAAC1GAHIx2x0gQEAYDkCkI+xFQYAANYjAPkYW2EAAGA9ApCPMQgaAADrVWkWWP369V02PL2QY8eOXVZBNV35cywj/wAAYJkqBaBp06aZPx89elTPPfecUlNTXTZDXblypcaNG+eVImsSPwZBAwBguSoFoCFDhpg///73v9df/vIXl81FH330Ub322mv69NNPXfblQkXlm6EyBggAAOu4PQZo5cqV6tmzZ4XjPXv21KeffuqRomqy8i4w1gECAMA6bgeghg0batmyZRWOL1u2TA0bNvRIUTUZW2EAAGA9t7fCmDhxoh544AGtXr1aSUlJkqR169ZpxYoVevPNNz1eYE3DOkAAAFjP7QA0dOhQtW7dWq++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mZn67rvvJEnffPONvvjiC/Xq1UsSz9lbPPVcs7KydNNNNykgIMC8JjU1Vbt27dLPP/98WTWyF5gPHDlyRGVlZS5/DCQpMjJSO3futKiqK5vT6dTo0aP1u9/9Tu3atZMk5eXlKSAgQOHh4S7XRkZGKi8vz7ymsv8fys/htIULF2rTpk36z3/+U+Ecz9kzfvzxR73++utKT0/X008/rf/85z969NFHFRAQoCFDhpjPqbLneO5zbty4sct5f39/NWjQgOd8xlNPPSWHw6FWrVrJz89PZWVlev755zVo0CBJ4jl7iaeea15enhISEirco/xc/fr1L7lGAhCuSCNGjNC2bdv0xRdfWF1KjbNv3z6NGjVKq1atUlBQkNXl1FhOp1NdunTRCy+8IEnq1KmTtm3bppkzZ2rIkCEWV1dz/OMf/9C7776r9957T23btlV2drZGjx6tmJgYnnMtRxeYD0RERMjPz6/CLJn8/HxFRUVZVNWVa+TIkfrwww/1+eefq0mTJubxqKgolZaWqqCgwOX6c59zVFRUpf8/lJ/D6S6uQ4cO6brrrpO/v7/8/f21Zs0avfrqq/L391dkZCTP2QOio6PVpk0bl2OtW7dWTk6OpLPP6UL/bkRFRenQoUMu50+dOqVjx47xnM944okn9NRTT2nAgAFq37697rvvPv3pT39SRkaGJJ6zt3jquXrz3xICkA8EBASoc+fOyszMNI85nU5lZmYqOTnZwsquLIZhaOTIkXr//ff12WefVWgW7dy5s6666iqX57xr1y7l5OSYzzk5OVlbt251+Y9u1apVCg0NrfDHqLa69dZbtXXrVmVnZ5uvLl26aNCgQebPPOfL97vf/a7CMg7fffed4uPjJUkJCQmKiopyec4Oh0Pr1q1zec4FBQXauHGjec1nn30mp9OppKQkH3yL6u+XX36R3e76p87Pz09Op1MSz9lbPPVck5OTtXbtWp08edK8ZtWqVWrZsuVldX9JYhq8ryxcuNAIDAw05s2bZ3z77bfGgw8+aISHh7vMksGFPfzww0ZYWJixevVqIzc313z98ssv5jUPPfSQERcXZ3z22WfGhg0bjOTkZCM5Odk8Xz49+/bbbzeys7ONFStWGI0aNWJ69kWcOwvMMHjOnrB+/XrD39/feP75543vv//eePfdd406deoY77zzjnnN5MmTjfDwcGPZsmXGli1bjL59+1Y6jbhTp07GunXrjC+++MJo0aJFrZ+efa4hQ4YYV199tTkNfsmSJUZERIQxZswY8xqe86U5fvy4sXnzZmPz5s2GJGPq1KnG5s2bjb179xqG4ZnnWlBQYERGRhr33XefsW3bNmPhwoVGnTp1mAZ/pfnrX/9qxMXFGQEBAUbXrl2Nr7/+2uqSriiSKn3NnTvXvObXX381HnnkEaN+/fpGnTp1jLvuusvIzc11uc9PP/1k9OrVywgODjYiIiKMxx57zDh58qSPv82V5bcBiOfsGR988IHRrl07IzAw0GjVqpUxa9Ysl/NOp9MYN26cERkZaQQGBhq33nqrsWvXLpdrjh49agwcONCoW7euERoaaqSlpRnHjx/35deo1hwOhzFq1CgjLi7OCAoKMpo1a2b8+c9/dplWzXO+NJ9//nml/yYPGTLEMAzPPddvvvnGuPHGG43AwEDj6quvNiZPnuyR+tkNHgAA1DqMAQIAALUOAQgAANQ6BCAAAFDrEIAAAECtQwACAAC1DgEIAADUOgQgAABQ6xCAAABArUMAAgAAtQ4BCEC1cvjwYQUEBKi4uFgnT55USEiIuUP6+Tz77LOy2WwVXq1atfJR1QCuNP5WFwAA58rKylJiYqJCQkK0bt06NWjQQHFxcRf9XNu2bfXpp5+6HPP35584AJWjBQhAtfLVV1/pd7/7nSTpiy++MH++GH9/f0VFRbm8IiIizPNNmzbVpEmTNHDgQIWEhOjqq6/WjBkzXO6Rk5Ojvn37qm7dugoNDdUf/vAH5efnu1zzwQcf6Prrr1dQUJAiIiJ01113mef+/ve/q0uXLqpXr56ioqL0xz/+UYcOHbrURwHAiwhAACyXk5Oj8PBwhYeHa+rUqXrjjTcUHh6up59+WkuXLlV4eLgeeeSRy/49L730khITE7V582Y99dRTGjVqlFatWiVJcjqd6tu3r44dO6Y1a9Zo1apV+vHHH9W/f3/z8x999JHuuusu3XHHHdq8ebMyMzPVtWtX8/zJkyc1adIkffPNN1q6dKl++uknDR069LLrBuB57AYPwHKnTp3S/v375XA41KVLF23YsEEhISHq2LGjPvroI8XFxalu3bouLTrnevbZZzVp0iQFBwe7HL/33ns1c+ZMSadbgFq3bq2PP/7YPD9gwAA5HA4tX75cq1atUq9evbRnzx7FxsZKkr799lu1bdtW69ev1/XXX69u3bqpWbNmeuedd6r0vTZs2KDrr79ex48fV926dS/l0QDwElqAAFjO399fTZs21c6dO3X99derQ4cOysvLU2RkpG666SY1bdr0vOGnXMuWLZWdne3y+stf/uJyTXJycoX3O3bskCTt2LFDsbGxZviRpDZt2ig8PNy8Jjs7W7feeut5a9i4caP69OmjuLg41atXTzfffLMkXXQQNwDfY4QgAMu1bdtWe/fu1cmTJ+V0OlW3bl2dOnVKp06dUt26dRUfH6/t27df8B4BAQFq3ry5V+v8bQvTuYqLi5WamqrU1FS9++67atSokXJycpSamqrS0lKv1gXAfbQAAbDc8uXLlZ2draioKL3zzjvKzs5Wu3btNG3aNGVnZ2v58uUe+T1ff/11hfetW7eWJLVu3Vr79u3Tvn37zPPffvutCgoK1KZNG0lShw4dlJmZWem9d+7cqaNHj2ry5Mnq3r27WrVqxQBooBqjBQiA5eLj45WXl6f8/Hz17dtXNptN27dv1+9//3tFR0dX6R6nTp1SXl6eyzGbzabIyEjz/ZdffqkpU6aoX79+WrVqlRYvXqyPPvpIkpSSkqL27dtr0KBBmjZtmk6dOqVHHnlEN998s7p06SJJmjBhgm699VZdc801GjBggE6dOqXly5frySefVFxcnAICAvTXv/5VDz30kLZt26ZJkyZ56AkB8DRagABUC6tXrzanl69fv15NmjSpcviRpO3btys6OtrlFR8f73LNY489pg0bNqhTp0567rnnNHXqVKWmpko6HZaWLVum+vXr66abblJKSoqaNWumRYsWmZ+/5ZZbtHjxYv3rX/9Sx44d9V//9V9av369JKlRo0aaN2+eFi9erDZt2mjy5Ml6+eWXPfBkAHgDs8AA1ApNmzbV6NGjNXr0aKtLAVAN0AIEAABqHQIQAACodegCAwAAtQ4tQAAAoNYhAAEAgFqHAAQAAGodAhAAAKh1CEAAAKDWIQABAIBahwAEAABqHQIQAACodQhAAACg1vn/LxT6OqP0JUgAAAAASUVORK5CYII=\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Predicción de conversión de bytes a kilobytes para un valor de 100 bytes\n",
        "print(\"Realizar una predicción!!!\")\n",
        "resultado = modelo.predict([100.0])\n",
        "print(\"El resultado es \" + str(resultado) + \" kilobytes!!\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "A6xbFLdWmC_-",
        "outputId": "9e22eb45-abad-4c7d-9bbf-a79da7c9be17"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Realizar una predicción!!!\n",
            "1/1 [==============================] - 0s 109ms/step\n",
            "El resultado es [[-0.61402994]] kilobytes!!\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Guardar el modelo entrenado\n",
        "modelo.save('types_of_computer_data.h5')"
      ],
      "metadata": {
        "id": "tVddfiLqmDbG",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "fdaf320f-e2aa-4328-a725-69093bd2981f"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.10/dist-packages/keras/src/engine/training.py:3079: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.\n",
            "  saving_api.save_model(\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!ls"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "-pLOnWgOmLJp",
        "outputId": "4ba1ddcb-679a-4e8f-89ea-ced8a1c8f6b6"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "sample_data  types_of_computer_data.h5\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install tensorflowjs"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        },
        "id": "q0bIj596mVHF",
        "outputId": "8e527e66-a599-49c6-a95f-742f14d74e10"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Collecting tensorflowjs\n",
            "  Downloading tensorflowjs-4.14.0-py3-none-any.whl (89 kB)\n",
            "\u001b[?25l     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m0.0/89.2 kB\u001b[0m \u001b[31m?\u001b[0m eta \u001b[36m-:--:--\u001b[0m\r\u001b[2K     \u001b[91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[91m╸\u001b[0m\u001b[90m━━━\u001b[0m \u001b[32m81.9/89.2 kB\u001b[0m \u001b[31m2.6 MB/s\u001b[0m eta \u001b[36m0:00:01\u001b[0m\r\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m89.2/89.2 kB\u001b[0m \u001b[31m2.2 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hRequirement already satisfied: flax>=0.7.2 in /usr/local/lib/python3.10/dist-packages (from tensorflowjs) (0.7.5)\n",
            "Requirement already satisfied: importlib_resources>=5.9.0 in /usr/local/lib/python3.10/dist-packages (from tensorflowjs) (6.1.1)\n",
            "Requirement already satisfied: jax>=0.4.13 in /usr/local/lib/python3.10/dist-packages (from tensorflowjs) (0.4.20)\n",
            "Requirement already satisfied: jaxlib>=0.4.13 in /usr/local/lib/python3.10/dist-packages (from tensorflowjs) (0.4.20+cuda11.cudnn86)\n",
            "Requirement already satisfied: tensorflow<3,>=2.13.0 in /usr/local/lib/python3.10/dist-packages (from tensorflowjs) (2.14.0)\n",
            "Collecting tensorflow-decision-forests>=1.5.0 (from tensorflowjs)\n",
            "  Downloading tensorflow_decision_forests-1.8.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (15.3 MB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m15.3/15.3 MB\u001b[0m \u001b[31m61.4 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hRequirement already satisfied: six<2,>=1.16.0 in /usr/local/lib/python3.10/dist-packages (from tensorflowjs) (1.16.0)\n",
            "Requirement already satisfied: tensorflow-hub>=0.14.0 in /usr/local/lib/python3.10/dist-packages (from tensorflowjs) (0.15.0)\n",
            "Requirement already satisfied: packaging~=23.1 in /usr/local/lib/python3.10/dist-packages (from tensorflowjs) (23.2)\n",
            "Requirement already satisfied: numpy>=1.22 in /usr/local/lib/python3.10/dist-packages (from flax>=0.7.2->tensorflowjs) (1.23.5)\n",
            "Requirement already satisfied: msgpack in /usr/local/lib/python3.10/dist-packages (from flax>=0.7.2->tensorflowjs) (1.0.7)\n",
            "Requirement already satisfied: optax in /usr/local/lib/python3.10/dist-packages (from flax>=0.7.2->tensorflowjs) (0.1.7)\n",
            "Requirement already satisfied: orbax-checkpoint in /usr/local/lib/python3.10/dist-packages (from flax>=0.7.2->tensorflowjs) (0.4.3)\n",
            "Requirement already satisfied: tensorstore in /usr/local/lib/python3.10/dist-packages (from flax>=0.7.2->tensorflowjs) (0.1.45)\n",
            "Requirement already satisfied: rich>=11.1 in /usr/local/lib/python3.10/dist-packages (from flax>=0.7.2->tensorflowjs) (13.7.0)\n",
            "Requirement already satisfied: typing-extensions>=4.2 in /usr/local/lib/python3.10/dist-packages (from flax>=0.7.2->tensorflowjs) (4.5.0)\n",
            "Requirement already satisfied: PyYAML>=5.4.1 in /usr/local/lib/python3.10/dist-packages (from flax>=0.7.2->tensorflowjs) (6.0.1)\n",
            "Requirement already satisfied: ml-dtypes>=0.2.0 in /usr/local/lib/python3.10/dist-packages (from jax>=0.4.13->tensorflowjs) (0.2.0)\n",
            "Requirement already satisfied: opt-einsum in /usr/local/lib/python3.10/dist-packages (from jax>=0.4.13->tensorflowjs) (3.3.0)\n",
            "Requirement already satisfied: scipy>=1.9 in /usr/local/lib/python3.10/dist-packages (from jax>=0.4.13->tensorflowjs) (1.11.4)\n",
            "Requirement already satisfied: absl-py>=1.0.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow<3,>=2.13.0->tensorflowjs) (1.4.0)\n",
            "Requirement already satisfied: astunparse>=1.6.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow<3,>=2.13.0->tensorflowjs) (1.6.3)\n",
            "Requirement already satisfied: flatbuffers>=23.5.26 in /usr/local/lib/python3.10/dist-packages (from tensorflow<3,>=2.13.0->tensorflowjs) (23.5.26)\n",
            "Requirement already satisfied: gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow<3,>=2.13.0->tensorflowjs) (0.5.4)\n",
            "Requirement already satisfied: google-pasta>=0.1.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow<3,>=2.13.0->tensorflowjs) (0.2.0)\n",
            "Requirement already satisfied: h5py>=2.9.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow<3,>=2.13.0->tensorflowjs) (3.9.0)\n",
            "Requirement already satisfied: libclang>=13.0.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow<3,>=2.13.0->tensorflowjs) (16.0.6)\n",
            "Requirement already satisfied: protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.20.3 in /usr/local/lib/python3.10/dist-packages (from tensorflow<3,>=2.13.0->tensorflowjs) (3.20.3)\n",
            "Requirement already satisfied: setuptools in /usr/local/lib/python3.10/dist-packages (from tensorflow<3,>=2.13.0->tensorflowjs) (67.7.2)\n",
            "Requirement already satisfied: termcolor>=1.1.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow<3,>=2.13.0->tensorflowjs) (2.3.0)\n",
            "Requirement already satisfied: wrapt<1.15,>=1.11.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow<3,>=2.13.0->tensorflowjs) (1.14.1)\n",
            "Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow<3,>=2.13.0->tensorflowjs) (0.34.0)\n",
            "Requirement already satisfied: grpcio<2.0,>=1.24.3 in /usr/local/lib/python3.10/dist-packages (from tensorflow<3,>=2.13.0->tensorflowjs) (1.59.3)\n",
            "Requirement already satisfied: tensorboard<2.15,>=2.14 in /usr/local/lib/python3.10/dist-packages (from tensorflow<3,>=2.13.0->tensorflowjs) (2.14.1)\n",
            "Requirement already satisfied: tensorflow-estimator<2.15,>=2.14.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow<3,>=2.13.0->tensorflowjs) (2.14.0)\n",
            "Requirement already satisfied: keras<2.15,>=2.14.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow<3,>=2.13.0->tensorflowjs) (2.14.0)\n",
            "Requirement already satisfied: pandas in /usr/local/lib/python3.10/dist-packages (from tensorflow-decision-forests>=1.5.0->tensorflowjs) (1.5.3)\n",
            "Collecting tensorflow<3,>=2.13.0 (from tensorflowjs)\n",
            "  Downloading tensorflow-2.15.0.post1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (475.2 MB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m475.2/475.2 MB\u001b[0m \u001b[31m1.2 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hRequirement already satisfied: wheel in /usr/local/lib/python3.10/dist-packages (from tensorflow-decision-forests>=1.5.0->tensorflowjs) (0.42.0)\n",
            "Collecting wurlitzer (from tensorflow-decision-forests>=1.5.0->tensorflowjs)\n",
            "  Downloading wurlitzer-3.0.3-py3-none-any.whl (7.3 kB)\n",
            "Collecting tensorboard<2.16,>=2.15 (from tensorflow<3,>=2.13.0->tensorflowjs)\n",
            "  Downloading tensorboard-2.15.1-py3-none-any.whl (5.5 MB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m5.5/5.5 MB\u001b[0m \u001b[31m85.2 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hCollecting tensorflow-estimator<2.16,>=2.15.0 (from tensorflow<3,>=2.13.0->tensorflowjs)\n",
            "  Downloading tensorflow_estimator-2.15.0-py2.py3-none-any.whl (441 kB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m442.0/442.0 kB\u001b[0m \u001b[31m43.5 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hCollecting keras<2.16,>=2.15.0 (from tensorflow<3,>=2.13.0->tensorflowjs)\n",
            "  Downloading keras-2.15.0-py3-none-any.whl (1.7 MB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m1.7/1.7 MB\u001b[0m \u001b[31m78.3 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hRequirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.10/dist-packages (from rich>=11.1->flax>=0.7.2->tensorflowjs) (3.0.0)\n",
            "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.10/dist-packages (from rich>=11.1->flax>=0.7.2->tensorflowjs) (2.16.1)\n",
            "Requirement already satisfied: google-auth<3,>=1.6.3 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.16,>=2.15->tensorflow<3,>=2.13.0->tensorflowjs) (2.17.3)\n",
            "Requirement already satisfied: google-auth-oauthlib<2,>=0.5 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.16,>=2.15->tensorflow<3,>=2.13.0->tensorflowjs) (1.0.0)\n",
            "Requirement already satisfied: markdown>=2.6.8 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.16,>=2.15->tensorflow<3,>=2.13.0->tensorflowjs) (3.5.1)\n",
            "Requirement already satisfied: requests<3,>=2.21.0 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.16,>=2.15->tensorflow<3,>=2.13.0->tensorflowjs) (2.31.0)\n",
            "Requirement already satisfied: tensorboard-data-server<0.8.0,>=0.7.0 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.16,>=2.15->tensorflow<3,>=2.13.0->tensorflowjs) (0.7.2)\n",
            "Requirement already satisfied: werkzeug>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.16,>=2.15->tensorflow<3,>=2.13.0->tensorflowjs) (3.0.1)\n",
            "Requirement already satisfied: chex>=0.1.5 in /usr/local/lib/python3.10/dist-packages (from optax->flax>=0.7.2->tensorflowjs) (0.1.7)\n",
            "Requirement already satisfied: etils[epath,epy] in /usr/local/lib/python3.10/dist-packages (from orbax-checkpoint->flax>=0.7.2->tensorflowjs) (1.5.2)\n",
            "Requirement already satisfied: nest_asyncio in /usr/local/lib/python3.10/dist-packages (from orbax-checkpoint->flax>=0.7.2->tensorflowjs) (1.5.8)\n",
            "Requirement already satisfied: python-dateutil>=2.8.1 in /usr/local/lib/python3.10/dist-packages (from pandas->tensorflow-decision-forests>=1.5.0->tensorflowjs) (2.8.2)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas->tensorflow-decision-forests>=1.5.0->tensorflowjs) (2023.3.post1)\n",
            "Requirement already satisfied: dm-tree>=0.1.5 in /usr/local/lib/python3.10/dist-packages (from chex>=0.1.5->optax->flax>=0.7.2->tensorflowjs) (0.1.8)\n",
            "Requirement already satisfied: toolz>=0.9.0 in /usr/local/lib/python3.10/dist-packages (from chex>=0.1.5->optax->flax>=0.7.2->tensorflowjs) (0.12.0)\n",
            "Requirement already satisfied: cachetools<6.0,>=2.0.0 in /usr/local/lib/python3.10/dist-packages (from google-auth<3,>=1.6.3->tensorboard<2.16,>=2.15->tensorflow<3,>=2.13.0->tensorflowjs) (5.3.2)\n",
            "Requirement already satisfied: pyasn1-modules>=0.2.1 in /usr/local/lib/python3.10/dist-packages (from google-auth<3,>=1.6.3->tensorboard<2.16,>=2.15->tensorflow<3,>=2.13.0->tensorflowjs) (0.3.0)\n",
            "Requirement already satisfied: rsa<5,>=3.1.4 in /usr/local/lib/python3.10/dist-packages (from google-auth<3,>=1.6.3->tensorboard<2.16,>=2.15->tensorflow<3,>=2.13.0->tensorflowjs) (4.9)\n",
            "Requirement already satisfied: requests-oauthlib>=0.7.0 in /usr/local/lib/python3.10/dist-packages (from google-auth-oauthlib<2,>=0.5->tensorboard<2.16,>=2.15->tensorflow<3,>=2.13.0->tensorflowjs) (1.3.1)\n",
            "Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.10/dist-packages (from markdown-it-py>=2.2.0->rich>=11.1->flax>=0.7.2->tensorflowjs) (0.1.2)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.21.0->tensorboard<2.16,>=2.15->tensorflow<3,>=2.13.0->tensorflowjs) (3.3.2)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.21.0->tensorboard<2.16,>=2.15->tensorflow<3,>=2.13.0->tensorflowjs) (3.6)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.21.0->tensorboard<2.16,>=2.15->tensorflow<3,>=2.13.0->tensorflowjs) (2.0.7)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.21.0->tensorboard<2.16,>=2.15->tensorflow<3,>=2.13.0->tensorflowjs) (2023.11.17)\n",
            "Requirement already satisfied: MarkupSafe>=2.1.1 in /usr/local/lib/python3.10/dist-packages (from werkzeug>=1.0.1->tensorboard<2.16,>=2.15->tensorflow<3,>=2.13.0->tensorflowjs) (2.1.3)\n",
            "Requirement already satisfied: fsspec in /usr/local/lib/python3.10/dist-packages (from etils[epath,epy]->orbax-checkpoint->flax>=0.7.2->tensorflowjs) (2023.6.0)\n",
            "Requirement already satisfied: zipp in /usr/local/lib/python3.10/dist-packages (from etils[epath,epy]->orbax-checkpoint->flax>=0.7.2->tensorflowjs) (3.17.0)\n",
            "Requirement already satisfied: pyasn1<0.6.0,>=0.4.6 in /usr/local/lib/python3.10/dist-packages (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard<2.16,>=2.15->tensorflow<3,>=2.13.0->tensorflowjs) (0.5.1)\n",
            "Requirement already satisfied: oauthlib>=3.0.0 in /usr/local/lib/python3.10/dist-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<2,>=0.5->tensorboard<2.16,>=2.15->tensorflow<3,>=2.13.0->tensorflowjs) (3.2.2)\n",
            "Installing collected packages: wurlitzer, tensorflow-estimator, keras, tensorboard, tensorflow, tensorflow-decision-forests, tensorflowjs\n",
            "  Attempting uninstall: tensorflow-estimator\n",
            "    Found existing installation: tensorflow-estimator 2.14.0\n",
            "    Uninstalling tensorflow-estimator-2.14.0:\n",
            "      Successfully uninstalled tensorflow-estimator-2.14.0\n",
            "  Attempting uninstall: keras\n",
            "    Found existing installation: keras 2.14.0\n",
            "    Uninstalling keras-2.14.0:\n",
            "      Successfully uninstalled keras-2.14.0\n",
            "  Attempting uninstall: tensorboard\n",
            "    Found existing installation: tensorboard 2.14.1\n",
            "    Uninstalling tensorboard-2.14.1:\n",
            "      Successfully uninstalled tensorboard-2.14.1\n",
            "  Attempting uninstall: tensorflow\n",
            "    Found existing installation: tensorflow 2.14.0\n",
            "    Uninstalling tensorflow-2.14.0:\n",
            "      Successfully uninstalled tensorflow-2.14.0\n",
            "Successfully installed keras-2.15.0 tensorboard-2.15.1 tensorflow-2.15.0.post1 tensorflow-decision-forests-1.8.1 tensorflow-estimator-2.15.0 tensorflowjs-4.14.0 wurlitzer-3.0.3\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "application/vnd.colab-display-data+json": {
              "pip_warning": {
                "packages": [
                  "keras",
                  "tensorboard",
                  "tensorflow"
                ]
              }
            }
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!mkdir types_of_computer_data"
      ],
      "metadata": {
        "id": "xdKOnZTNmWDm"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "!tensorflowjs_converter --input_format keras types_of_computer_data.h5 types_of_computer_data"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "NsIeZOf9mXee",
        "outputId": "aaea9fb4-0a8f-4ac6-ca13-2b8b15da73d1"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "2023-12-06 02:40:35.172200: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered\n",
            "2023-12-06 02:40:35.172284: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered\n",
            "2023-12-06 02:40:35.174234: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered\n",
            "2023-12-06 02:40:37.859459: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!ls types_of_computer_data"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "TeIEtt94mYgX",
        "outputId": "e30655dc-d655-4f40-c7da-b6f783b27700"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "group1-shard1of1.bin  model.json\n"
          ]
        }
      ]
    }
  ]
}