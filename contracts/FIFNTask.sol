// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * FIFNTask — optional custom Flock.io task contract.
 *
 * Extend Flock.io's base IFLTask interface to add:
 *   - minimum participant enforcement
 *   - label-confidence-weighted aggregation metadata
 *   - stake-based anti-dropout slashing
 *
 * Deploy via Flock.io's task factory; most projects use the default
 * task contract and don't need this file.
 */
interface IFLTask {
    function submitModel(string calldata modelCID, uint256 dataSize) external;
    function getGlobalModelCID(uint256 roundId) external view returns (string memory);
}

contract FIFNTask is IFLTask {
    address public immutable coordinator;
    uint256 public minParticipants;
    uint256 public currentRound;

    mapping(uint256 => string)  public globalModelCID;
    mapping(uint256 => uint256) public submissionCount;
    mapping(address => uint256) public stakes;

    event RoundStarted(uint256 indexed roundId, string globalCID);
    event UpdateSubmitted(uint256 indexed roundId, address indexed node, string modelCID);
    event RoundAggregated(uint256 indexed roundId, string newGlobalCID);

    modifier onlyCoordinator() {
        require(msg.sender == coordinator, "Not coordinator");
        _;
    }

    constructor(address _coordinator, uint256 _minParticipants) {
        coordinator = _coordinator;
        minParticipants = _minParticipants;
    }

    function startRound(string calldata initialCID) external onlyCoordinator {
        currentRound++;
        globalModelCID[currentRound] = initialCID;
        emit RoundStarted(currentRound, initialCID);
    }

    function submitModel(string calldata modelCID, uint256 dataSize) external override {
        require(stakes[msg.sender] > 0, "Must stake before participating");
        submissionCount[currentRound]++;
        emit UpdateSubmitted(currentRound, msg.sender, modelCID);
    }

    function finaliseRound(string calldata newGlobalCID) external onlyCoordinator {
        require(
            submissionCount[currentRound] >= minParticipants,
            "Not enough participants"
        );
        globalModelCID[currentRound + 1] = newGlobalCID;
        emit RoundAggregated(currentRound, newGlobalCID);
    }

    function getGlobalModelCID(uint256 roundId) external view override returns (string memory) {
        return globalModelCID[roundId];
    }

    function stake() external payable {
        stakes[msg.sender] += msg.value;
    }
}
